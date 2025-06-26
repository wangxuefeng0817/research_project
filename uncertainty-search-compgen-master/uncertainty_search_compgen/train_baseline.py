import sys
import os
import gc

project_root = "/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master"
if project_root not in sys.path:
    sys.path.append(project_root)

os.chdir("/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master")
sys.path.append("/home/wangx36/.local/lib/python3.11/site-packages")

if __name__ == "__main__":
    __package__ = "uncertainty_search_compgen"

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Config, AutoConfig
from .train_lm import T5Module
from .dataset import TokenizerPairIterableDataset
from .load_hf_lm import load_hf_tokenizer
from .data import load_smcalflow_cs_simplified
import pathlib
import argparse


def train_baseline_model(
    model_name_or_path="Salesforce/codet5p-220m",
    data_path="/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/text/semparse/smcalflow-cs",
    output_dir="logs_baseline",
    max_steps=10000,
    batch_size=52
):
    """训练只有标准T5的基准模型（没有ATS头）"""
    
    print("🎯 训练基准模型（没有adaptive temperature head）")
    print(f"   模型: {model_name_or_path}")
    print(f"   输出目录: {output_dir}")
    print(f"   最大步数: {max_steps}")
    
    # 设置随机种子
    pl.seed_everything(0)
    
    # 加载数据
    basepath = pathlib.Path(data_path)
    train_pairs, _, val_pairs, _ = load_smcalflow_cs_simplified(basepath)
    tokenizer, _ = load_hf_tokenizer(model_name_or_path)
    
    print(f"📊 数据统计:")
    print(f"   训练样本: {len(train_pairs)}")
    print(f"   验证样本: {len(val_pairs)}")
    
    # 加载模型
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False
    )
    model = T5ForConditionalGeneration.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir="/tmp/hf"
    )
    
    # 创建基准模型（train_mode="t5"表示不使用ATS头）
    harness = T5Module(
        model, 
        pad_token=tokenizer.pad_token_id,
        hidden_size=model.config.d_model,
        train_mode="t5"  # 🔑 关键：只训练标准T5，不使用ATS头
    )
    
    print("✅ 模型初始化完成（ATS头已冻结）")
    
    # 设置训练器
    torch.set_float32_matmul_precision("medium")
    
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
        print("🚀 使用GPU训练")
    else:
        accelerator = "cpu"
        devices = 1
        precision = "32"
        print("🐌 使用CPU训练")
    
    trainer = pl.Trainer(
        max_steps=max_steps,
        val_check_interval=500,
        num_sanity_val_steps=1,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        accumulate_grad_batches=1,
        enable_checkpointing=False,
        default_root_dir=output_dir
    )
    
    # 开始训练
    print("🏃‍♂️ 开始训练基准模型...")
    trainer.fit(
        harness,
        DataLoader(
            TokenizerPairIterableDataset(list(train_pairs), tokenizer),
            batch_size=batch_size,
        ),
        DataLoader(
            TokenizerPairIterableDataset(list(val_pairs), tokenizer), 
            batch_size=batch_size
        ),
    )
    
    # 保存模型
    output_path = f"{output_dir}/baseline_t5_model.ckpt"
    torch.save({
        "state_dict": harness.state_dict(),
        "pytorch-lightning_version": "2.2.2"
    }, output_path)
    
    print(f"✅ 基准模型训练完成！")
    print(f"📁 模型保存至: {output_path}")
    
    return trainer, model, harness


def main():
    parser = argparse.ArgumentParser(description="训练基准T5模型（不使用ATS头）")
    parser.add_argument("--model_name_or_path", type=str, default="Salesforce/codet5p-220m")
    parser.add_argument("--data_path", type=str, 
                        default="/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/text/semparse/smcalflow-cs")
    parser.add_argument("--output_dir", type=str, default="logs_baseline")
    parser.add_argument("--max_steps", type=int, default=10000, help="训练步数")
    parser.add_argument("--batch_size", type=int, default=52)
    
    args = parser.parse_args()
    
    # 创建输出目录
    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    # 开始训练
    train_baseline_model(
        model_name_or_path=args.model_name_or_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main() 