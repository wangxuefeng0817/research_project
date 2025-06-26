#!/usr/bin/env python3
"""
在完整的smcalflow-cs数据集上训练ATS模型（方案2.5）
"""

import sys
import os
sys.path.append('.') # 将当前目录添加到Python路径

from uncertainty_search_compgen.train_lm import finetune_tf_lm
from uncertainty_search_compgen.data import load_smcalflow_cs_simplified
from uncertainty_search_compgen.load_hf_lm import load_hf_tokenizer # 使用你的tokenizer加载函数

def main():
    print("🚀 开始在完整smcalflow-cs数据上训练ATS模型 (方案2.5)...")
    print("目标：大幅提升温度-损失相关性，冲击0.5+ 💪\n")
    
    # 1. 加载数据和tokenizer
    print("加载smcalflow-cs数据集...")
    (train_pairs, train_full, val_pairs, test_pairs) = load_smcalflow_cs_simplified(
        "text/semparse/smcalflow-cs" # 使用相对路径
    )
    print(f"训练数据 (train_pairs): {len(train_pairs)} 对")
    print(f"完整训练数据 (train_full): {len(train_full)} 对")
    print(f"验证数据 (val_pairs): {len(val_pairs)} 对")
    print(f"测试数据 (test_pairs): {len(test_pairs)} 对")
    
    print("\n加载codet5p-220m tokenizer...")
    tokenizer, _ = load_hf_tokenizer("Salesforce/codet5p-220m") # 使用你的tokenizer加载函数
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer pad token设置为eos_token: {tokenizer.eos_token}")

    # 准备ATS头训练数据 (train_full)
    # 注意：在你的笔记本中，train_data_for_temperature_head是如何定义的？
    # 这里我假设train_full就是用于ATS头训练的数据
    # 如果不同，请告诉我如何准备 train_data_for_temperature_head
    train_data_for_ats_head = train_full
    print(f"ATS头训练数据 (train_full): {len(train_data_for_ats_head)} 对")
    
    # 2. 开始训练
    print("\n🔥 开始方案2.5训练 (使用完整数据和codet5p-220m)... ")
    print("模型配置:")
    print("  ✅ 基础模型: Salesforce/codet5p-220m")
    print("  ✅ ATShead架构: 2层Transformer + 2层MLP (GELU, LayerNorm)")
    print("  ✅ 温度正则化: MSE对齐 + 温度分离 (权重1.5)")
    print("  ✅ 学习率: 主模型0.1x, ATS头1.0x")
    print("  ✅ 数据集: 完整的smcalflow-cs")
    
    # 注意：你笔记本中的 finetune_tf_lm 调用了 train_pairs[:-128] 或 [:-72]
    # 这里我们使用完整的 train_pairs 进行Stage 1训练
    # Stage 2 (ATS) 训练将使用 train_data_for_ats_head (即train_full)
    trainer, model, harness = finetune_tf_lm(
        model_name_or_path="Salesforce/codet5p-220m",
        tokenizer=tokenizer,
        train_pairs=train_pairs, # Stage 1 使用完整训练集
        train_full=train_data_for_ats_head, # Stage 2 (ATS) 使用完整训练集
        val_pairs=val_pairs,
        load_from=None,
        do_fit=True,
        root_dir="logs_full_data_stable"
    )
    
    print("\n🎉🎉🎉 方案2.5在完整数据上训练完成！🎉🎉🎉")
    print("现在可以运行测试脚本，使用新模型进行评估:")
    print("  python test_ats_improvements.py")
    print("\n期待相关性大幅提升！🚀")

if __name__ == "__main__":
    main() 