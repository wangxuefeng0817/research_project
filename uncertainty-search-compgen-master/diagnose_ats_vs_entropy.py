#!/usr/bin/env python3
"""
诊断ATS头和entropy在beam search中的行为差异
分析为什么ATS头在correlation测试中表现良好，但在beam search中效果不如entropy
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, '/scratch/work/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master')

from uncertainty_search_compgen.train_lm import T5Module
from uncertainty_search_compgen.load_hf_lm import load_hf_tokenizer
from uncertainty_search_compgen.data import load_smcalflow_cs_simplified
from uncertainty_search_compgen.dataset import TokenizerPairDataset
from torch.utils.data import DataLoader
from transformers import T5Config, T5ForConditionalGeneration
import torch.nn.functional as F

def load_model_and_tokenizer(checkpoint_path, train_mode="ats"):
    """加载模型和tokenizer"""
    config = T5Config.from_pretrained("Salesforce/codet5p-220m", 
                                     output_hidden_states=True, 
                                     return_dict=True, 
                                     use_cache=False)
    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5p-220m", 
                                                      config=config, 
                                                      cache_dir="/tmp/hf")
    tokenizer, _ = load_hf_tokenizer("Salesforce/codet5p-220m")
    
    harness = T5Module(model, 
                      pad_token=tokenizer.pad_token_id, 
                      hidden_size=model.config.d_model, 
                      train_mode=train_mode)
    
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        harness.load_state_dict(torch.load(checkpoint_path)["state_dict"], strict=False)
    
    if torch.cuda.is_available():
        harness.cuda()
    
    return harness, tokenizer

@torch.no_grad()
def analyze_single_step_predictions(harness, batch, tokenizer, max_steps=5):
    """分析单步预测中ATS头和entropy的表现"""
    harness.eval()
    
    inp_batch = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()
    batch_size = inp_batch.shape[0]
    
    # 初始化decoder input
    tgt = torch.ones_like(inp_batch[:, :1]) * harness.model.config.decoder_start_token_id
    
    step_results = []
    
    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        
        # 1. 获取ATS头的预测
        if harness.train_mode == "ats":
            temp_batch = {
                "input_ids": inp_batch,
                "labels": tgt
            }
            scaled_logits, ats_temperatures, original_logits = harness.forward(temp_batch)
            next_token_logits = scaled_logits[:, -1, :]  # 最后一个位置的logits
            
            # 计算ATS头预测的"uncertainty"（这里用temperature的倒数作为uncertainty的proxy）
            if hasattr(ats_temperatures, 'mean'):
                ats_uncertainty = ats_temperatures[:, -1, :].mean(dim=-1)  # 取平均温度
            else:
                ats_uncertainty = torch.ones(batch_size, device=inp_batch.device) * ats_temperatures
                
            print(f"ATS temperatures shape: {ats_temperatures.shape if hasattr(ats_temperatures, 'shape') else 'scalar'}")
            print(f"ATS uncertainty (avg temp): {ats_uncertainty.mean().item():.4f}")
        else:
            # 标准模型预测
            output = harness.model(
                input_ids=inp_batch,
                decoder_input_ids=tgt,
                attention_mask=torch.ones_like(inp_batch),
                decoder_attention_mask=torch.ones_like(tgt),
            )
            next_token_logits = output.logits[:, -1, :]
            ats_uncertainty = torch.ones(batch_size, device=inp_batch.device)
        
        # 2. 计算entropy
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        entropy = torch.special.entr(next_token_probs).sum(dim=-1)
        
        print(f"Entropy: {entropy.mean().item():.4f}")
        
        # 3. 计算top-k的概率分布
        top_k_probs, top_k_indices = next_token_probs.topk(5, dim=-1)
        
        # 4. 分析预测的一致性
        # 检查ATS头的uncertainty和entropy是否在相同的token上给出相似的判断
        if harness.train_mode == "ats":
            # 计算correlation
            if len(ats_uncertainty.shape) == 1 and len(entropy.shape) == 1:
                correlation = torch.corrcoef(torch.stack([ats_uncertainty, entropy]))[0, 1]
                print(f"ATS-Entropy correlation: {correlation.item():.4f}")
            else:
                print(f"Shape mismatch: ATS {ats_uncertainty.shape}, Entropy {entropy.shape}")
        
        # 5. 保存结果
        step_result = {
            'step': step,
            'ats_uncertainty': ats_uncertainty.cpu().numpy(),
            'entropy': entropy.cpu().numpy(),
            'top_k_probs': top_k_probs.cpu().numpy(),
            'top_k_indices': top_k_indices.cpu().numpy(),
        }
        step_results.append(step_result)
        
        # 6. 选择下一个token（取概率最高的）
        next_token = next_token_probs.argmax(dim=-1)
        tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
        
        # 检查是否结束
        if (next_token == tokenizer.eos_token_id).all():
            break
    
    return step_results

@torch.no_grad()
def compare_beam_search_guidance(harness, batch, tokenizer, threshold=0.4):
    """比较ATS头和entropy在beam search guidance中的表现"""
    harness.eval()
    
    inp_batch = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()
    batch_size = inp_batch.shape[0]
    
    # 初始化
    tgt = torch.ones_like(inp_batch[:, :1]) * harness.model.config.decoder_start_token_id
    
    print(f"\n=== Beam Search Guidance Comparison ===")
    print(f"Threshold: {threshold}")
    
    for step in range(5):  # 只分析前5步
        print(f"\n--- Step {step + 1} ---")
        
        # 1. 获取logits和ATS预测
        if harness.train_mode == "ats":
            temp_batch = {
                "input_ids": inp_batch,
                "labels": tgt
            }
            scaled_logits, ats_temperatures, original_logits = harness.forward(temp_batch)
            out_logits = original_logits[:, -1:, :]  # 使用原始logits计算entropy
            
            # 计算ATS uncertainty
            if hasattr(ats_temperatures, 'mean'):
                ats_uncertainty = ats_temperatures[:, -1, :].mean(dim=-1)
            else:
                ats_uncertainty = torch.ones(batch_size, device=inp_batch.device) * ats_temperatures
        else:
            output = harness.model(
                input_ids=inp_batch,
                decoder_input_ids=tgt,
                attention_mask=torch.ones_like(inp_batch),
                decoder_attention_mask=torch.ones_like(tgt),
            )
            out_logits = output.logits[:, -1:, :]
            ats_uncertainty = torch.ones(batch_size, device=inp_batch.device)
        
        # 2. 计算entropy
        out_logits_p = out_logits.softmax(dim=-1)
        entropy = torch.special.entr(out_logits_p).sum(dim=-1).squeeze()
        
        # 3. 比较两种guidance的决策
        # ATS guidance: 基于temperature
        ats_expand = ats_uncertainty > threshold
        
        # Entropy guidance: 基于entropy
        entropy_expand = entropy > threshold
        
        # 4. 统计一致性
        agreement = (ats_expand == entropy_expand).float().mean()
        
        print(f"ATS uncertainty: {ats_uncertainty.mean().item():.4f} (expand: {ats_expand.sum().item()}/{batch_size})")
        print(f"Entropy: {entropy.mean().item():.4f} (expand: {entropy_expand.sum().item()}/{batch_size})")
        print(f"Agreement: {agreement.item():.4f}")
        
        # 5. 分析disagreement的cases
        disagreement_mask = ats_expand != entropy_expand
        if disagreement_mask.sum() > 0:
            print(f"Disagreement cases: {disagreement_mask.sum().item()}")
            for i in range(batch_size):
                if disagreement_mask[i]:
                    ats_val = ats_uncertainty[i].item() if ats_uncertainty[i].numel() > 0 else ats_uncertainty.item()
                    entropy_val = entropy[i].item() if entropy[i].numel() > 0 else entropy.item()
                    print(f"  Sample {i}: ATS={ats_val:.4f}->{'expand' if ats_expand[i] else 'keep'}, "
                          f"Entropy={entropy_val:.4f}->{'expand' if entropy_expand[i] else 'keep'}")
        
        # 6. 前进到下一步
        next_token = out_logits_p.argmax(dim=-1)
        tgt = torch.cat([tgt, next_token], dim=1)
        
        if (next_token.squeeze() == tokenizer.eos_token_id).all():
            break
    
    return {
        'ats_uncertainty_mean': ats_uncertainty.mean().item(),
        'entropy_mean': entropy.mean().item(),
        'agreement': agreement.item()
    }

def run_diagnostic_analysis():
    """运行完整的诊断分析"""
    basepath = Path("/scratch/work/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master")
    
    # 1. 加载数据
    print("Loading data...")
    _, _, _, test_pairs = load_smcalflow_cs_simplified(
        basepath / "text/semparse/smcalflow-cs"
    )
    tokenizer, _ = load_hf_tokenizer("Salesforce/codet5p-220m")
    
    # 准备小样本数据
    small_dataset = TokenizerPairDataset(test_pairs[:5], tokenizer)
    dataloader = DataLoader(small_dataset, batch_size=2, shuffle=False)
    
    # 2. 加载ATS模型
    print("Loading ATS model...")
    checkpoint_path = basepath / "logs_full_data_stable/stage2/final_model_with_new_tempurature_stage2.ckpt"
    ats_model, _ = load_model_and_tokenizer(str(checkpoint_path), train_mode="ats")
    
    # 3. 加载基础模型（用于对比）
    print("Loading base model...")
    base_model, _ = load_model_and_tokenizer(None, train_mode="t5")
    
    # 4. 运行分析
    print("\n" + "="*60)
    print("DIAGNOSTIC ANALYSIS: ATS vs Entropy")
    print("="*60)
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n### Batch {batch_idx + 1} ###")
        
        # 移动到GPU
        batch = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in batch.items()}
        
        print("\n1. Single Step Analysis (ATS Model):")
        step_results = analyze_single_step_predictions(ats_model, batch, tokenizer)
        
        print("\n2. Beam Search Guidance Comparison:")
        guidance_results = compare_beam_search_guidance(ats_model, batch, tokenizer)
        
        # 保存结果
        results = {
            'batch_idx': batch_idx,
            'step_results': step_results,
            'guidance_results': guidance_results
        }
        
        with open(basepath / f"diagnostic_results_batch_{batch_idx}.json", 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
        
        if batch_idx >= 2:  # 只分析前3个batch
            break
    
    print("\n" + "="*60)
    print("Analysis completed! Results saved to diagnostic_results_batch_*.json")
    print("="*60)

if __name__ == "__main__":
    run_diagnostic_analysis()