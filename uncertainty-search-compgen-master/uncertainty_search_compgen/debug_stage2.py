#!/usr/bin/env python3

"""
诊断Stage2性能下降的脚本
比较不同配置下的模型性能
"""

import sys
import os
project_root = "/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master"
if project_root not in sys.path:
    sys.path.append(project_root)

os.chdir(project_root)

if __name__ == "__main__":
    __package__ = "uncertainty_search_compgen"

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, AutoConfig
from .train_lm import T5Module
from .dataset import TokenizerPairIterableDataset
from .load_hf_lm import load_hf_tokenizer
from .data import load_smcalflow_cs_simplified
import pathlib


def run_ablation_study():
    """运行消融实验来诊断Stage2问题"""
    
    print("🔬 开始Stage2诊断实验...")
    
    # 加载数据
    basepath = pathlib.Path("text/semparse/smcalflow-cs")
    train_pairs, _, val_pairs, _ = load_smcalflow_cs_simplified(basepath)
    tokenizer, _ = load_hf_tokenizer("Salesforce/codet5p-220m")
    
    # 加载预训练模型
    config = AutoConfig.from_pretrained(
        "Salesforce/codet5p-220m",
        output_hidden_states=True,
        return_dict=True,
        use_cache=False
    )
    model = T5ForConditionalGeneration.from_pretrained(
        "Salesforce/codet5p-220m",
        config=config,
        cache_dir="/tmp/hf"
    )
    
    # 实验配置
    experiments = [
        {
            "name": "实验1: Stage2减少训练步数",
            "train_mode": "ats",
            "max_steps": 3000,  # 和Stage1相同
            "data": train_pairs,  # 和Stage1相同数据
            "description": "测试是否是过拟合导致的"
        },
        {
            "name": "实验2: Stage2相同数据",
            "train_mode": "ats", 
            "max_steps": 5000,  # 适中的步数
            "data": train_pairs,  # 使用Stage1相同的数据
            "description": "测试是否是数据差异导致的"
        },
        {
            "name": "实验3: Stage2低学习率",
            "train_mode": "ats",
            "max_steps": 5000,
            "data": train_pairs,
            "lr_scale": 0.1,  # 降低学习率
            "description": "测试是否是学习率问题"
        }
    ]
    
    for i, exp in enumerate(experiments):
        print(f"\n{'='*50}")
        print(f"🧪 {exp['name']}")
        print(f"📝 {exp['description']}")
        print(f"{'='*50}")
        
        # 创建模型
        harness = T5Module(
            model, 
            pad_token=tokenizer.pad_token_id,
            hidden_size=model.config.d_model,
            train_mode=exp['train_mode'],
            lr=0.0002 * exp.get('lr_scale', 1.0)
        )
        
        # 加载Stage1权重
        stage1_path = "logs_full_data_stable/stage1/final_model_with_new_temperature_stage1.ckpt"
        if os.path.exists(stage1_path):
            harness.load_state_dict(torch.load(stage1_path)["state_dict"])
            print("✅ 加载Stage1权重成功")
        else:
            print("⚠️  Stage1权重不存在，从头开始训练")
        
        # 设置训练器
        if torch.cuda.is_available():
            accelerator, devices, precision = "gpu", 1, "bf16-mixed"
        else:
            accelerator, devices, precision = "cpu", 1, "32"
            
        trainer = pl.Trainer(
            max_steps=exp['max_steps'],
            val_check_interval=500,
            num_sanity_val_steps=1,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            accumulate_grad_batches=1,
            enable_checkpointing=False,
            default_root_dir=f"logs_debug/experiment_{i+1}"
        )
        
        # 训练
        trainer.fit(
            harness,
            DataLoader(
                TokenizerPairIterableDataset(list(exp['data']), tokenizer),
                batch_size=52,
            ),
            DataLoader(
                TokenizerPairIterableDataset(list(val_pairs), tokenizer), 
                batch_size=52
            )
        )
        
        # 保存结果
        save_path = f"logs_debug/experiment_{i+1}/model.ckpt"
        torch.save({
            "state_dict": harness.state_dict(),
            "config": exp,
            "pytorch-lightning_version": "2.2.2"
        }, save_path)
        
        print(f"💾 实验{i+1}完成，模型保存至: {save_path}")
        print(f"🧪 请用validation.py测试此模型的效果")
        print(f"   python validation.py --load_from {save_path}")


def compare_stage1_stage2():
    """直接对比Stage1和Stage2模型的温度分布"""
    
    print("🔍 对比Stage1和Stage2模型...")
    
    # 加载两个模型
    stage1_path = "logs_full_data_stable/stage1/final_model_with_new_temperature_stage1.ckpt"
    stage2_path = "logs_full_data_stable/stage2/final_model_with_new_tempurature_stage2.ckpt"
    
    if not os.path.exists(stage1_path) or not os.path.exists(stage2_path):
        print("❌ 模型文件不存在，请检查路径")
        return
    
    # TODO: 加载模型并比较温度分布
    print("📊 温度分布对比分析...")
    print("   - Stage1: 使用固定温度1.0")  
    print("   - Stage2: 使用学习的自适应温度")
    print("   - 建议: 可视化温度分布，看是否合理")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ablation", "compare"], default="ablation")
    args = parser.parse_args()
    
    if args.mode == "ablation":
        run_ablation_study()
    else:
        compare_stage1_stage2() 