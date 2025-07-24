#!/usr/bin/env python3
"""
测试改进的ATS beam search方法性能
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, '/scratch/work/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master')

from uncertainty_search_compgen.validation import run_evaluation
from uncertainty_search_compgen.train_lm import T5Module
from uncertainty_search_compgen.load_hf_lm import load_hf_tokenizer
from transformers import T5Config, T5ForConditionalGeneration

def test_improved_ats_methods():
    """测试改进的ATS方法"""
    basepath = Path("/scratch/work/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master")
    
    # 加载ATS模型
    print("🔄 Loading ATS model...")
    config = T5Config.from_pretrained("Salesforce/codet5p-220m", 
                                     output_hidden_states=True, 
                                     return_dict=True, 
                                     use_cache=False)
    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5p-220m", 
                                                      config=config, 
                                                      cache_dir="/tmp/hf")
    
    tokenizer, _ = load_hf_tokenizer("Salesforce/codet5p-220m")
    
    # 创建ATS模型
    harness = T5Module(model, 
                      pad_token=tokenizer.pad_token_id, 
                      hidden_size=model.config.d_model, 
                      train_mode="ats")
    
    # 加载checkpoint
    checkpoint_path = basepath / "logs_full_data_stable/stage2/final_model_with_new_tempurature_stage2.ckpt"
    if checkpoint_path.exists():
        print(f"🔄 Loading checkpoint: {checkpoint_path}")
        harness.load_state_dict(torch.load(str(checkpoint_path))["state_dict"], strict=False)
    else:
        print("⚠️  No checkpoint found, using untrained model")
    
    if torch.cuda.is_available():
        harness.cuda()
    
    # 确保eval_scores目录存在
    (basepath / "eval_scores").mkdir(exist_ok=True)
    
    # 运行评估
    print("\n" + "="*60)
    print("🚀 Testing Improved ATS Methods")
    print("="*60)
    
    run_evaluation(
        model=harness,
        epoch=0,
        run_name="improved_ats_methods_test",
        num_samples=20  # 先用小样本测试
    )
    
    print("\n✅ Testing completed!")
    print("📊 Check eval_scores/ directory for results")

if __name__ == "__main__":
    test_improved_ats_methods()