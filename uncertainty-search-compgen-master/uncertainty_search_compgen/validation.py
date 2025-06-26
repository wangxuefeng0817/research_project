import os, sys
import pickle
import pathlib
from functools import partial
from collections import defaultdict
from time import perf_counter

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
import pytorch_lightning as pl
from transformers import T5Config, T5ForConditionalGeneration

# Set up env
os.environ["HF_HOME"] = "/tmp/hf"
basepath = pathlib.Path(os.environ.get("WRKDIR", "."))
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from uncertainty_search_compgen.train_lm import T5Module
from uncertainty_search_compgen.dataset import TokenizerPairDataset
from uncertainty_search_compgen.load_hf_lm import load_hf_tokenizer
from uncertainty_search_compgen.data import load_smcalflow_cs_simplified
from uncertainty_search_compgen.inference_ats import (
    beam_search_hf,
    get_topk_outputs,
    simple_temperature_test,
    entropy_beam_search,
    uncertainty_guided_search,
    temperature_rerank_beam_search,
)

@torch.no_grad()
def compute_accuracy_by_batch_by_k(preds, targets, pad_target_idx):
    """
    Compute accuracy and exact matches for a batch.

    Args:
        preds (torch.Tensor): Predicted tokens (BATCH, SEQ_LEN).
        targets (torch.Tensor): Target tokens (BATCH, SEQ_LEN, @K).
        pad_target_idx (int): Padding token index in targets.

    Returns:
        dict: Contains token-level accuracy and exact matches for the batch.
    """
    
    # Get dimensions
    batch_size, pred_len, k = preds.shape
    target_len = targets.shape[1]
    
    # Truncate to minimum length to avoid shape mismatch
    min_len = min(pred_len, target_len)
    preds_truncated = preds[:, :min_len, :]  # (BATCH, MIN_LEN, K)
    targets_truncated = targets[:, :min_len]  # (BATCH, MIN_LEN)
    
    # Repeat the predictions for the last dimension
    targets_rep = targets_truncated.unsqueeze(-1).repeat(1, 1, k)

    # Compute the mask
    actions_mask = targets_truncated == pad_target_idx
    
    # 1. truncate the output for each pair
    truncated_matches = []
    for ap, mask, tgt in zip(preds_truncated, actions_mask, targets_rep):
        # ap: (MIN_LEN, K), mask: (MIN_LEN,), tgt: (MIN_LEN, K)
        valid_mask = ~mask  # (MIN_LEN,)
        if valid_mask.sum() > 0:
            # Extract valid positions for both ap and tgt
            valid_ap = ap[valid_mask]  # (VALID_LEN, K)
            valid_tgt = tgt[valid_mask]  # (VALID_LEN, K)
            matches = (valid_ap == valid_tgt)  # (VALID_LEN, K)
            truncated_matches.append(matches)
        else:
            # If no valid tokens, create a dummy match tensor
            truncated_matches.append(torch.ones((1, k), dtype=torch.bool))

    # 2. calculate the accuracy and exacts for each pair. 
    # truncated_matches: (batch, seq_len, k)

    acc = [x.sum(dim=0).div(x.shape[0]).max().item() for x in truncated_matches]
    exacts = [tm.all(dim=0).max().item() for tm in truncated_matches]

    return {
        "acc": acc,
        "exacts": exacts,
    }


@torch.inference_mode()
def compute_validation_metrics(
    harness,
    val_pairs,
    sampler,
    verbose=True,
    return_results=False,
):

    # Collect metrics manually
    device = harness.device
    pad_target_idx = harness.hparams.pad_token
    val_accs = []
    val_exacts = []
    outs = []
    
    for batch in tqdm(val_pairs, desc="Evaluating", leave=False):
        # Move batch to GPU
        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}

        # Calculate the completion
        # shape: (BATCH, SEQ_LEN, @K)
        out_tokens = sampler(harness, batch).detach().cpu()
        
        if return_results:
            outs.extend(out_tokens.transpose(1, 2).tolist()) # shape: (B, K, SEQ)

        # Compute stats
        stats = compute_accuracy_by_batch_by_k(out_tokens, batch["labels"].cpu(), pad_target_idx)

        val_accs.extend(stats['acc'])
        val_exacts.extend(stats['exacts'])

    if verbose:
        print(f"Accuracy    : {np.mean(val_accs):.4f}")
        print(f"Exact Match : {np.mean(val_exacts):.4f}")

    return {"accuracy": val_accs, "exacts": val_exacts, "results": outs}


def uncertainty_guided_search_wrapper(
    harness, batch, tokenizer, k=32, keep_n=3, out_length=64, **kwargs
):
    """
    包装uncertainty_guided_search，转换输出格式为标准格式
    Return shape: (BATCH, SEQ_LEN, @K)
    """
    # 调用原始的uncertainty_guided_search
    # 注意：uncertainty_guided_search内部会+1，所以我们不需要再+1
    out = uncertainty_guided_search(
        harness,
        batch,
        tokenizer,
        k=k,
        keep_n=keep_n,
        out_length=out_length,  # 🔥 修复：不再+1
        **kwargs,
    )
    out_logits = []
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    for b in out:
        _batch = []
        for beam in b[:keep_n]:
            # 🔥 更安全的处理方式
            tgt_seq = beam[1][1:]  # tgt，移除start token
            # 确保长度不超过期望长度
            if len(tgt_seq) > out_length:
                tgt_seq = tgt_seq[:out_length]
            _batch.append(tgt_seq)
        
        # 确保有足够的beams
        while len(_batch) < keep_n:
            _batch.append(np.full(out_length, pad_token_id, dtype=np.int64))

        # 确保所有序列长度一致
        for i in range(len(_batch)):
            if len(_batch[i]) < out_length:
                # Pad到指定长度
                padding_length = out_length - len(_batch[i])
                _batch[i] = np.concatenate([_batch[i], np.full(padding_length, pad_token_id, dtype=np.int64)])
            elif len(_batch[i]) > out_length:
                # 截断到指定长度
                _batch[i] = _batch[i][:out_length]

        out_logits.append(np.vstack(_batch))

    out_logits = torch.tensor(np.array(out_logits), dtype=torch.int64).permute(
        (0, 2, 1)
    )

    return out_logits


@torch.inference_mode()
def run_evaluation(model, epoch, run_name, num_samples=50):
    basepath = pathlib.Path(os.environ.get("WRKDIR", "."))

    _, _, _, test_pairs = load_smcalflow_cs_simplified(
        basepath / "/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/text/semparse/smcalflow-cs"
    )
    tokenizer, _ = load_hf_tokenizer("Salesforce/codet5p-220m")
    
    # 根据参数决定样本数量和batch size
    total_samples = len(test_pairs)
    samples_to_use = min(num_samples, total_samples)
    
    # 动态调整batch size
    if samples_to_use <= 50:
        batch_size = 16
    elif samples_to_use <= 200:
        batch_size = 8
    else:
        batch_size = 4
    
    dataloader = DataLoader(
        TokenizerPairDataset(test_pairs[:samples_to_use], tokenizer), 
        batch_size=batch_size, 
        shuffle=False
    )

    # 测试温度重排序beam search的效果（高效版本）
    samplers = {
        # 🔥 标准HuggingFace beam search（基准）
        'hf_standard': partial(beam_search_hf, beams=5, k=5, early_stopping=True, max_length=128),
        
        # 🔥 温度重排序beam search - 不同alpha值（高效版本）
        'temp_rerank_α0.0': partial(temperature_rerank_beam_search, beams=5, k=5, alpha=0.0, max_length=128),
        'temp_rerank_α0.3': partial(temperature_rerank_beam_search, beams=5, k=5, alpha=0.3, max_length=128),
        'temp_rerank_α0.5': partial(temperature_rerank_beam_search, beams=5, k=5, alpha=0.5, max_length=128),
        'temp_rerank_α0.8': partial(temperature_rerank_beam_search, beams=5, k=5, alpha=0.8, max_length=128),
        'temp_rerank_α1.0': partial(temperature_rerank_beam_search, beams=5, k=5, alpha=1.0, max_length=128),
        
        # 🔥 对比：其他搜索方法
        'entropy_search': partial(entropy_beam_search, tokenizer=tokenizer, steps=64, keep_n=5, out_length=128),
        'uncertainty_search': partial(uncertainty_guided_search_wrapper, tokenizer=tokenizer, k=64, keep_n=5, out_length=128),
    }

    print(f"\n{'='*60}")
    print(f"🔬 运行评估: {run_name} (epoch {epoch})")
    print(f"📊 数据集: {samples_to_use}/{total_samples} 样本")
    print(f"📦 批次大小: {batch_size}")
    print(f"🎯 模型训练模式: {model.train_mode}")
    print(f"{'='*60}")

    for name, sampler in samplers.items():
        print(f"\n📋 方法: {name}")
        
        stats = compute_validation_metrics(
            harness=model,
            val_pairs=dataloader,
            sampler=sampler,
            return_results=True,
        )

        with open(
            basepath / f"eval_scores/{run_name}_epoch{epoch}_{samples_to_use}samples_{name}.pickle", "wb"
        ) as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✅ 评估完成！结果保存到 eval_scores/ (文件名包含样本数量)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run validation using original code")
    parser.add_argument("--load_from", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--run_name", type=str, default="clean_validation")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--disable_temperature", action="store_true", help="强制禁用温度缩放")
    parser.add_argument("--num_samples", type=int, default=50, help="要测试的样本数量 (默认: 50)")
    args = parser.parse_args()
    
    # Load model
    config = T5Config.from_pretrained("Salesforce/codet5p-220m", output_hidden_states=True, return_dict=True, use_cache=False)
    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5p-220m", config=config, cache_dir="/tmp/hf")
    
    tokenizer, _ = load_hf_tokenizer("Salesforce/codet5p-220m")
    
    # 根据参数和checkpoint确定训练模式
    if args.disable_temperature:
        train_mode = "t5"
        print("🧪 实验模式: 强制禁用温度缩放")
    else:
        # 根据checkpoint路径自动判断
        if "ats" in args.load_from.lower() or "stage2" in args.load_from.lower():
            train_mode = "ats"
        else:
            train_mode = "t5"
    
    harness = T5Module(model, pad_token=tokenizer.pad_token_id, hidden_size=model.config.d_model, train_mode=train_mode)
    
    print(f"🔄 加载checkpoint: {args.load_from}")
    harness.load_state_dict(torch.load(args.load_from)["state_dict"], strict=False)
    
    # 确保训练模式设置正确
    if args.disable_temperature:
        harness.train_mode = "t5"  
    
    if torch.cuda.is_available():
        harness.cuda()

    # Create eval_scores directory
    basepath = pathlib.Path(os.environ.get("WRKDIR", "."))
    (basepath / "eval_scores").mkdir(exist_ok=True)
    
    # Run evaluation
    run_evaluation(harness, args.epoch, args.run_name, args.num_samples)


if __name__ == "__main__":
    main()