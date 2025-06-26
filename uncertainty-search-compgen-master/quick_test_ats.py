#!/usr/bin/env python3
"""
快速测试ATS改进效果的训练脚本
"""

import sys
import os
sys.path.append('.')

from uncertainty_search_compgen.train_lm import finetune_tf_lm
from transformers import T5Tokenizer
import json

def create_test_data():
    """创建测试数据"""
    # 简单的数学问题，用于测试
    test_pairs = [
        ("What is 2 + 3?", "5"),
        ("What is 5 - 2?", "3"),
        ("What is 4 * 2?", "8"),
        ("What is 10 / 2?", "5"),
        ("What is 3 + 4?", "7"),
        ("What is 8 - 3?", "5"),
        ("What is 3 * 3?", "9"),
        ("What is 12 / 3?", "4"),
        ("What is 6 + 2?", "8"),
        ("What is 9 - 1?", "8"),
    ] * 10  # 重复10次增加数据量
    
    return test_pairs, test_pairs[:50], test_pairs[:20]  # train, train_full, val

def main():
    print("开始快速测试ATS改进效果...")
    
    # 1. 准备数据和tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_pairs, train_full, val_pairs = create_test_data()
    print(f"训练数据: {len(train_pairs)} 对")
    print(f"验证数据: {len(val_pairs)} 对")
    
    # 2. 开始训练
    print("\n开始两阶段训练...")
    trainer, model, harness = finetune_tf_lm(
        model_name_or_path='t5-small',
        tokenizer=tokenizer,
        train_pairs=train_pairs,
        train_full=train_full,
        val_pairs=val_pairs,
        load_from=None,
        do_fit=True,
        root_dir="logs_test"
    )
    
    print("\n训练完成！现在可以运行测试脚本:")
    print("python test_ats_improvements.py")

if __name__ == "__main__":
    main() 