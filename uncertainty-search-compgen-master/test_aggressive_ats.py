#!/usr/bin/env python3
"""
测试ATS方案3激进优化效果
目标：达到0.5以上的温度-损失相关性
"""

import sys
import os
sys.path.append('.')

from uncertainty_search_compgen.train_lm import finetune_tf_lm
from transformers import T5Tokenizer
import json

def create_diverse_test_data():
    """创建更多样化的测试数据"""
    # 数学问题（简单到复杂）
    math_data = [
        ("What is 1 + 1?", "2"),
        ("What is 5 * 3?", "15"),
        ("What is 17 + 28?", "45"),
        ("What is 144 / 12?", "12"),
        ("What is 23 * 17?", "391"),
        ("What is the square root of 64?", "8"),
        ("What is 15% of 200?", "30"),
        ("What is 2^8?", "256"),
    ]
    
    # 常识问题（简单到难）
    knowledge_data = [
        ("What color is the sun?", "Yellow"),
        ("How many legs does a cat have?", "Four"),
        ("What is the capital of France?", "Paris"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("In what year did World War II end?", "1945"),
        ("What is the chemical symbol for gold?", "Au"),
        ("Who wrote Romeo and Juliet?", "William Shakespeare"),
    ]
    
    # 翻译任务（简单到复杂）
    translation_data = [
        ("Translate 'Hello' to French", "Bonjour"),
        ("Translate 'Thank you' to Spanish", "Gracias"),
        ("Translate 'Good morning' to German", "Guten Morgen"),
        ("Translate 'How are you?' to Italian", "Come stai?"),
        ("Translate 'I love you' to Japanese", "Aishiteru"),
        ("Translate 'Beautiful' to Russian", "Krasivaya"),
        ("Translate 'Friendship' to Portuguese", "Amizade"),
        ("Translate 'Understanding' to Chinese", "理解"),
    ]
    
    # 逻辑推理（简单到复杂）
    reasoning_data = [
        ("If all birds can fly, and a robin is a bird, can a robin fly?", "Yes"),
        ("If it's raining, the ground gets wet. It's raining. Is the ground wet?", "Yes"),
        ("John is taller than Mary. Mary is taller than Sue. Who is the shortest?", "Sue"),
        ("If A > B and B > C, what is the relationship between A and C?", "A > C"),
        ("All roses are flowers. Some flowers are red. Are all roses red?", "No"),
        ("If today is Monday, what day was it three days ago?", "Friday"),
        ("A train leaves at 2 PM and arrives at 5 PM. How long is the journey?", "3 hours"),
        ("If x + 5 = 12, what is x?", "7"),
    ]
    
    # 组合所有数据，重复多次
    all_data = (math_data + knowledge_data + translation_data + reasoning_data) * 3
    
    return all_data, all_data[:len(all_data)//2], all_data[:len(all_data)//4]

def main():
    print("🚀 开始测试ATS方案3激进优化效果...")
    print("目标：达到0.5以上的温度-损失相关性\n")
    
    # 1. 准备数据和tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_pairs, train_full, val_pairs = create_diverse_test_data()
    print(f"训练数据: {len(train_pairs)} 对")
    print(f"完整训练数据: {len(train_full)} 对")
    print(f"验证数据: {len(val_pairs)} 对")
    
    # 2. 开始训练
    print("\n🔥 开始方案3激进优化训练...")
    print("改进点:")
    print("  ✅ 增强ATShead架构 (4层 + 自注意力 + 残差)")
    print("  ✅ 强化温度正则化 (权重2.0 + 多策略)")
    print("  ✅ 同时训练主模型和ATS头")
    print("  ✅ 差异化学习率 (主模型0.5x, ATS头2.0x)")
    
    trainer, model, harness = finetune_tf_lm(
        model_name_or_path='t5-small',
        tokenizer=tokenizer,
        train_pairs=train_pairs,
        train_full=train_full,
        val_pairs=val_pairs,
        load_from=None,
        do_fit=True,
        root_dir="logs_aggressive"
    )
    
    print("\n🎯 方案3激进优化训练完成！")
    print("现在可以运行测试脚本验证效果:")
    print("python test_ats_improvements.py")
    print("\n期待相关性提升到0.5以上！🎊")

if __name__ == "__main__":
    main() 