#!/usr/bin/env python
# coding: utf-8
# %%

# %%



# %%


import torch
print("Torch file path:", torch.__file__)


# %%


import os
os.environ["HF_HOME"] = "/tmp/hf"


# %%


import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))


# %%


import os
print(os.getcwd())


# %%


import sys
sys.path.append(os.path.abspath("/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master"))


# %%


import sys
print(sys.executable)


# %%


import sys
print(sys.path)


# %%


import sys
sys.path.append("/home/wangx36/.local/lib/python3.11/site-packages")


# %%


from uncertainty_search_compgen.data import load_smcalflow_cs_simplified
from uncertainty_search_compgen.load_hf_lm import load_hf_tokenizer
from uncertainty_search_compgen.train_lm import finetune_tf_lm,T5Module
from uncertainty_search_compgen.inference import get_topk_outputs, generate_gpt, uncertainty_guided_search
from uncertainty_search_compgen.dataset import TokenizerPairDataset
from uncertainty_search_compgen.divergence_metrics import (
    measure_entropy,
    measure_mutual_kl_causal_mask,
    measure_teacher_student_model_divergence
)
from uncertainty_search_compgen.plotting import visualize_as_table, plot_batch_and_metric, plot_batch_and_metric_pair

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from uncertainty_search_compgen.data import format_source, retokenize_input,parse_json_objects


# %%


(
    train_pairs,
    train_full,
    val_pairs,
    test_pairs
) = load_smcalflow_cs_simplified(
    "/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/text/semparse/smcalflow-cs"
)


# %%


train_pairs[:1]


# %%


# Fewshot data
train_full[-128:-126]


# %%


test_pairs[:2]


# %%


tokenizer, idx2word = load_hf_tokenizer("Salesforce/codet5p-220m")


# %%


def create_combined_pytorch_dataset(dataset1, dataset2, ratio1=0.5, ratio2=0.5, shuffle=True):
    """
    从两个PyTorch数据集中按指定比例抽取样本，组成一个新的数据集
    
    参数:
    dataset1: 第一个PyTorch数据集
    dataset2: 第二个PyTorch数据集
    ratio1: 从第一个数据集中抽取的比例（默认0.5，即50%）
    ratio2: 从第二个数据集中抽取的比例（默认0.5，即50%）
    shuffle: 是否打乱样本顺序（默认True）
    
    返回:
    combined_dataset: 合并后的PyTorch数据集
    """
    import random
    import pickle
    from torch.utils.data import Subset, ConcatDataset
    
    # 计算要抽取的样本数量
    sample_count1 = int(len(dataset1) * ratio1)
    sample_count2 = int(len(dataset2) * ratio2)
    
    # 生成索引列表
    indices1 = list(range(len(dataset1)))
    indices2 = list(range(len(dataset2)))
    
    # 随机打乱索引（如果需要）
    if shuffle:
        random.shuffle(indices1)
        random.shuffle(indices2)
    
    # 抽取子集
    selected_indices1 = indices1[:sample_count1]
    selected_indices2 = indices2[:sample_count2]
    
    subset1 = Subset(dataset1, selected_indices1)
    subset2 = Subset(dataset2, selected_indices2)
    
    # 合并数据集
    combined_dataset = ConcatDataset([subset1, subset2])
    
    print(f"从数据集1中抽取了{len(subset1)}个样本")
    print(f"从数据集2中抽取了{len(subset2)}个样本")
    print(f"合并后的数据集包含{len(combined_dataset)}个样本")

    return combined_dataset




def load_jsonl(file_path, split_label):
    with open(file_path) as f:
        return [{**o, "split": split_label} for o in parse_json_objects(f.readlines())]



high_loss_samples = load_jsonl('/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/high_loss_dataset.jsonl', 'high_loss')



def process_data(data):
        return [
            (retokenize_input(x["source"]), retokenize_input(x["target"]), x["qid"])
            for x in data if x["target"] is not None
        ]



high_loss_samples = process_data(high_loss_samples)



train_data_for_temperature_head = create_combined_pytorch_dataset(train_full, high_loss_samples, ratio1=0.1, ratio2=1, shuffle=True)


# %%


# Note: This trains only the decoder (see transformer_optimizer_config)
#
# You might want to train the encoder as well, but for now, only the decoder.
train_only_trainer, train_only_model, train_only_harness = finetune_tf_lm(
    "Salesforce/codet5p-220m",
    tokenizer,
    train_pairs[:-128],
    train_full,
    val_pairs,
    root_dir="logs/train_only"
)


# %%


shot_trainer, shot_model, shot_harness = finetune_tf_lm(
    "Salesforce/codet5p-220m",
    tokenizer,
    train_pairs[:-72],# First 32 pairs
    train_full,
    val_pairs,
    root_dir="logs/shot_train"
)


# %%


# # 重用已有的trainer和model
# shot_trainer, shot_model, shot_harness = finetune_tf_lm(
#     "Salesforce/codet5p-220m",
#     tokenizer,
#     train_pairs[:-72],
#     train_full,
#     val_pairs,
#     load_from="logs/shot_train/stage2/final_model.ckpt",  # 加载保存的模型
#     do_fit=False  # 不重新训练
# )


# %%


# train_only_trainer, train_only_model, train_only_harness = finetune_tf_lm(
#     "Salesforce/codet5p-220m",
#     tokenizer,
#     train_pairs[:-128],
#     train_full,
#     val_pairs,
#     load_from="logs/train_only/stage2/final_model.ckpt",  # 加载training only的模型
#     do_fit=False  # 不重新训练
# )


# %%


train_only_trainer, train_only_model1, train_only_harness = finetune_tf_lm(
    "Salesforce/codet5p-220m",
    tokenizer,
    train_pairs[:-128],
    train_full,
    val_pairs,
    load_from="logs/train_only/stage1/final_model_for_generate_data.ckpt",  # 加载training only的模型
    do_fit=False  # 不重新训练
)


# %%


def compute_final_high_loss_samples(model, dataloader, pad_token, top_ratio, device, output_file='high_loss_qids.txt'):
    import torch.nn.functional as F
    
    loss_dict = {}
    qid_dict = {}  # 存储sample_id到qid的映射
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 确保模型在正确的设备上
            model = model.to(device)
            
            # 创建一个新的batch字典，所有张量都移动到正确的设备
            device_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    device_batch[k] = v.to(device)
                else:
                    device_batch[k] = v
            
            # 存储当前批次的qid
            if 'qid' in batch:
                for i, qid in enumerate(batch['qid']):
                    sample_id = batch_idx * dataloader.batch_size + i
                    qid_dict[sample_id] = qid
            
            # 使用device_batch进行模型操作
            outputs = model(
                input_ids=device_batch["input_ids"],
                attention_mask=device_batch.get("attention_mask", None),
                decoder_input_ids=device_batch.get("decoder_input_ids", None),
                decoder_attention_mask=device_batch.get("decoder_attention_mask", None),
                labels=device_batch.get("labels", None),
            )
            
            # 获取预测结果
            preds = outputs.logits
            
            # 计算token级别的损失（忽略pad tokens）
            token_losses = F.cross_entropy(
                preds.view(-1, preds.size(-1)),
                device_batch["labels"].view(-1),
                reduction='none',
                ignore_index=pad_token
            ).view(preds.shape[0], -1)
            
            # 存储batch_idx和计算的损失
            for i in range(batch["input_ids"].shape[0]):
                sample_id = batch_idx * dataloader.batch_size + i
                loss_dict[sample_id] = token_losses[i].mean().item()
    
    # 找出高损失样本
    sorted_losses = sorted(loss_dict.items(), key=lambda x: x[1], reverse=True)
    num_high_loss = int(len(sorted_losses) * top_ratio)
    high_loss_samples = [idx for idx, _ in sorted_losses[:num_high_loss]]
    
    # 收集高损失样本的qid
    high_loss_qids = []
    for idx in high_loss_samples:
        if idx in qid_dict:
            high_loss_qids.append(qid_dict[idx])
        else:
            high_loss_qids.append(str(idx))  # 如果没有qid，使用索引作为标识
    
    # 将qids保存到文件
    with open(output_file, 'w') as f:
        for qid in high_loss_qids:
            f.write(f"{qid}\n")
    
    print(f"已将{len(high_loss_qids)}个高损失样本的qid保存到 {output_file}")
    
    return high_loss_samples, high_loss_qids


# %%


dataset = TokenizerPairDataset(train_full, tokenizer)


# %%


train_dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
)


# %%


import torch
compute_final_high_loss_samples(train_only_model1, dataloader=train_dataloader, pad_token=tokenizer.pad_token_id, top_ratio=0.1, device="cuda")


# %%


def create_high_loss_dataset_from_qids(original_dataset_file, high_loss_qids_file, output_file):
    """
    根据高损失样本的qid文件从原始数据集中提取样本，创建一个新的数据集
    
    参数:
    original_dataset_file: 原始数据集的文件路径（JSONL格式）
    high_loss_qids_file: 包含高损失样本qid的文件路径
    output_file: 输出新数据集的文件路径
    
    返回:
    high_loss_samples_count: 提取的高损失样本数量
    """
    import json
    
    # 读取高损失样本的qid
    with open(high_loss_qids_file, 'r') as f:
        high_loss_qids = set([line.strip() for line in f])
    
    print(f"读取了{len(high_loss_qids)}个高损失样本的qid")
    
    # 从原始数据集中提取高损失样本
    high_loss_samples = []
    
    with open(original_dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                sample = json.loads(line.strip())
                if sample.get('qid') in high_loss_qids:
                    high_loss_samples.append(sample)
            except json.JSONDecodeError:
                print(f"警告: 跳过无效的JSON行: {line[:50]}...")
    
    print(f"从原始数据集中提取了{len(high_loss_samples)}个高损失样本")
    
    # 保存新数据集
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in high_loss_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"高损失样本数据集已保存到 {output_file}")
    
    return len(high_loss_samples)


# %%


create_high_loss_dataset_from_qids(
    original_dataset_file="/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/text/semparse/smcalflow-cs/",
    high_loss_qids_file="high_loss_qids.txt",
    output_file="high_loss_dataset.jsonl"
)


# %%


def create_combined_pytorch_dataset(dataset1, dataset2, ratio1=0.5, ratio2=0.5, shuffle=True,json_path=None):
    """
    从两个PyTorch数据集中按指定比例抽取样本，组成一个新的数据集
    
    参数:
    dataset1: 第一个PyTorch数据集
    dataset2: 第二个PyTorch数据集
    ratio1: 从第一个数据集中抽取的比例（默认0.5，即50%）
    ratio2: 从第二个数据集中抽取的比例（默认0.5，即50%）
    shuffle: 是否打乱样本顺序（默认True）
    
    返回:
    combined_dataset: 合并后的PyTorch数据集
    """
    import random
    import pickle
    from torch.utils.data import Subset, ConcatDataset
    
    # 计算要抽取的样本数量
    sample_count1 = int(len(dataset1) * ratio1)
    sample_count2 = int(len(dataset2) * ratio2)
    
    # 生成索引列表
    indices1 = list(range(len(dataset1)))
    indices2 = list(range(len(dataset2)))
    
    # 随机打乱索引（如果需要）
    if shuffle:
        random.shuffle(indices1)
        random.shuffle(indices2)
    
    # 抽取子集
    selected_indices1 = indices1[:sample_count1]
    selected_indices2 = indices2[:sample_count2]
    
    subset1 = Subset(dataset1, selected_indices1)
    subset2 = Subset(dataset2, selected_indices2)
    
    # 合并数据集
    combined_dataset = ConcatDataset([subset1, subset2])
    
    print(f"从数据集1中抽取了{len(subset1)}个样本")
    print(f"从数据集2中抽取了{len(subset2)}个样本")
    print(f"合并后的数据集包含{len(combined_dataset)}个样本")
    if json_path:
        # 使用pickle保存原始数据
        with open(json_path, 'wb') as f:
            pickle.dump(combined_dataset, f)
        
        print(f"数据集已保存到: {json_path}")
        print(combined_dataset[:2])
    return combined_dataset


# %%


def load_jsonl(file_path, split_label):
    with open(file_path) as f:
        return [{**o, "split": split_label} for o in parse_json_objects(f.readlines())]


# %%


high_loss_samples = load_jsonl('/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/high_loss_dataset.jsonl', 'high_loss')


# %%


def process_data(data):
        return [
            (retokenize_input(x["source"]), retokenize_input(x["target"]), x["qid"])
            for x in data if x["target"] is not None
        ]


# %%


high_loss_samples = process_data(high_loss_samples)


# %%


train_data_for_temperature_head = create_combined_pytorch_dataset(train_full, high_loss_samples, ratio1=0.05, ratio2=0.5, shuffle=True,json_path='/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/head.pkl')


# %%


single_test_example = next(iter(DataLoader(TokenizerPairDataset(test_pairs, tokenizer), batch_size=1)))


# %%


single_test_example_topk_outputs = get_topk_outputs(shot_harness, single_test_example)


# %%


# Metric 1: Token-level entropy
single_test_example_entropy = measure_entropy(shot_harness, single_test_example)


# %%


# Metric 2: Token-level entropy, using model uncertainty
single_test_example_model_entropy = measure_mutual_kl_causal_mask(shot_harness, single_test_example)


# %%


# Metric 3: Student-teacher KL divergence
single_test_example_student_teacher_divergence = measure_teacher_student_model_divergence(
    train_only_harness,
    shot_harness,
    single_test_example
)


# %%


print(shot_harness
     )


# %%


# # Loss
# with torch.inference_mode():
#     single_test_example_train_only_loss = F.cross_entropy(
#         train_only_harness.forward({
#             k: v.to(shot_harness.device)
#             for k, v in single_test_example.items()
#         }).flatten(0, -2),
#         single_test_example["labels"].to(shot_harness.device).flatten(),
#         reduction="none"
#     )[None].cpu()
#     single_test_example_shot_loss = F.cross_entropy(
#         shot_harness.forward({
#             k: v.to(shot_harness.device)
#             for k, v in single_test_example.items()
#         }).flatten(0, -2),
#         single_test_example["labels"].to(shot_harness.device).flatten(),
#         reduction="none"
#     )[None].cpu()


# %%


# Loss
with torch.inference_mode():
    output = train_only_harness.forward({
        k: v.to(shot_harness.device)
        for k, v in single_test_example.items()
    })
    logits = output[0] 
    

    single_test_example_train_only_loss = F.cross_entropy(
        logits.flatten(0, -2), 
        single_test_example["labels"].to(shot_harness.device).flatten(),
        reduction="none"
    )[None].cpu()
    output2 = shot_harness.forward({
            k: v.to(shot_harness.device)
            for k, v in single_test_example.items()
        })[0]
    single_test_example_shot_loss = F.cross_entropy(
        output2.flatten(0, -2),
        single_test_example["labels"].to(shot_harness.device).flatten(),
        reduction="none"
    )[None].cpu()


# %%


# Topk outputs
single_test_example_topk = get_topk_outputs(shot_harness, single_test_example)


# %%


visualize_as_table(
    single_test_example_entropy,
    single_test_example_student_teacher_divergence,
    single_test_example_train_only_loss,
    single_test_example_shot_loss,
    single_test_example_topk,
    single_test_example,
    idx2word,
    shot_harness.hparams.pad_token
)


# %%


plot_batch_and_metric(
    tokenizer,
    single_test_example,
    single_test_example_shot_loss,
    idx2word,
    shot_harness.hparams.pad_token
)


# %%


plot_batch_and_metric_pair(
    tokenizer,
    single_test_example,
    single_test_example_shot_loss,
    single_test_example_student_teacher_divergence,
    idx2word,
    shot_harness.hparams.pad_token
)


# %%


get_ipython().run_line_magic('pdb', 'on')


# %%


# These are all of the options that get generated by the uncertainty search
[
    [
        tokenizer.batch_decode(y, skip_special_tokens=True)
        for y in x
    ]
    for x in
    uncertainty_guided_search(
        shot_harness,
        single_test_example,
        tokenizer,
    )
]


# %%


import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1️⃣ 加载 embedding 模型
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2️⃣ 生成 5 个文档的 embedding
documents = ["人工智能", "深度学习", "机器学习", "计算机视觉", "自然语言处理"]
embeddings = model.encode(documents)  # shape = (5, 384)
embeddings = np.array(embeddings, dtype=np.float32)  # FAISS 需要 float32 格式

# 3️⃣ 创建 FAISS 索引
index = faiss.IndexFlatL2(embeddings.shape[1])  # 选择 L2 距离索引
print(f"索引维度: {index.d}")  # 输出索引的维度
print(f"索引初始大小: {index.ntotal}")  # 查看索引中的向量数

# 4️⃣ 添加向量到索引
index.add(embeddings)

# 5️⃣ 查看索引大小
print(f"索引添加后的大小: {index.ntotal}")  # 现在索引应该存有 5 个向量


# %%





# %%




