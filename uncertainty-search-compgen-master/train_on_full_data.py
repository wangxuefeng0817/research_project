#!/usr/bin/env python3
"""
Train ATS model on complete smcalflow-cs dataset (Approach 2.5)
"""

import sys
import os
sys.path.append('.') # Add current directory to Python path

from uncertainty_search_compgen.train_lm import finetune_tf_lm
from uncertainty_search_compgen.data import load_smcalflow_cs_simplified
from uncertainty_search_compgen.load_hf_lm import load_hf_tokenizer # Use your tokenizer loading function

def main():
    print("🚀 Starting ATS model training on complete smcalflow-cs data (Approach 2.5)...")
    print("Goal: Significantly improve temperature-loss correlation, targeting 0.5+ 💪\n")
    
    # 1. Load data and tokenizer
    print("Loading smcalflow-cs dataset...")
    (train_pairs, train_full, val_pairs, test_pairs) = load_smcalflow_cs_simplified(
        "text/semparse/smcalflow-cs" # Use relative path
    )
    print(f"Training data (train_pairs): {len(train_pairs)} pairs")
    print(f"Full training data (train_full): {len(train_full)} pairs")
    print(f"Validation data (val_pairs): {len(val_pairs)} pairs")
    print(f"Test data (test_pairs): {len(test_pairs)} pairs")
    
    print("\nLoading codet5p-220m tokenizer...")
    tokenizer, _ = load_hf_tokenizer("Salesforce/codet5p-220m") # Use your tokenizer loading function
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer pad token set to eos_token: {tokenizer.eos_token}")

    # Prepare ATS head training data (train_full)
    # Note: How is train_data_for_temperature_head defined in your notebook?
    # Here I assume train_full is the data used for ATS head training
    # If different, please tell me how to prepare train_data_for_temperature_head
    train_data_for_ats_head = train_full
    print(f"ATS head training data (train_full): {len(train_data_for_ats_head)} pairs")
    
    # 2. Start training
    print("\n🔥 Starting Approach 2.5 training (using complete data and codet5p-220m)... ")
    print("Model configuration:")
    print("  ✅ Base model: Salesforce/codet5p-220m")
    print("  ✅ ATShead architecture: 2-layer Transformer + 2-layer MLP (GELU, LayerNorm)")
    print("  ✅ Temperature regularization: MSE alignment + temperature separation (weight 1.5)")
    print("  ✅ Learning rate: Main model 0.1x, ATS head 1.0x")
    print("  ✅ Dataset: Complete smcalflow-cs")
    
    # Note: Your notebook's finetune_tf_lm called train_pairs[:-128] or [:-72]
    # Here we use complete train_pairs for Stage 1 training
    # Stage 2 (ATS) training will use train_data_for_ats_head (i.e., train_full)
    trainer, model, harness = finetune_tf_lm(
        model_name_or_path="Salesforce/codet5p-220m",
        tokenizer=tokenizer,
        train_pairs=train_pairs, # Stage 1 uses complete training set
        train_full=train_data_for_ats_head, # Stage 2 (ATS) uses complete training set
        val_pairs=val_pairs,
        load_from=None,
        do_fit=True,
        root_dir="logs_full_data_stable"
    )
    
    print("\n🎉🎉🎉 Approach 2.5 training on complete data completed! 🎉🎉🎉")
    print("Now you can run test scripts to evaluate using the new model:")
    print("  python test_ats_improvements.py")
    print("\nExpecting significant correlation improvement! 🚀")

if __name__ == "__main__":
    main() 