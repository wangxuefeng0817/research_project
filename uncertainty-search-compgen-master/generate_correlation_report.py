#!/usr/bin/env python3
"""
生成ATS头层数与不确定性度量相关性的对比报告。

该脚本会：
1. 遍历不同层数（1, 2, 3, 4）的Stage-2 ATS模型。
2. 对每个模型，计算其预测温度与Stage-1基线模型损失的皮尔逊相关系数。
3. 计算一次词元熵与Stage-1损失的相关性作为基线。
4. 生成一个条形图，直观对比不同层数ATS模型和词元熵的效果。
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings

from uncertainty_search_compgen.load_hf_lm import load_hf_tokenizer
from uncertainty_search_compgen.train_lm import T5Module
from transformers import T5ForConditionalGeneration, AutoConfig

warnings.filterwarnings('ignore')

def get_correlation_scores(ats_model, stage1_model, tokenizer, device="cuda"):
    """
    为给定的模型计算不确定性度量与损失的相关系数。
    返回 (ats_temp_correlation, entropy_correlation)
    """
    ats_model.eval()
    stage1_model.eval()

    test_samples = [
        {"input": "What is the capital of France?", "output": "Paris"},
        {"input": "Translate 'hello' to German.", "output": "Hallo"},
        {"input": "Write a Python function for factorial.", "output": "def factorial(n): return 1 if n == 0 else n * factorial(n-1)"},
    ] * 30 # 使用少量但重复的样本进行快速评估

    all_ats_temperatures = []
    all_stage1_losses = []
    all_stage1_entropies = []

    with torch.no_grad():
        for sample in test_samples:
            inputs = tokenizer(sample['input'], max_length=256, truncation=True, padding='max_length', return_tensors='pt')
            labels = tokenizer(sample['output'], max_length=256, truncation=True, padding='max_length', return_tensors='pt')['input_ids']

            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            inputs['labels'] = labels

            try:
                # Get ATS temperature
                _, ats_temperatures, _ = ats_model(inputs)

                # Get Stage1 loss and entropy
                start_token = stage1_model.model.config.decoder_start_token_id or tokenizer.pad_token_id
                decoder_input_ids = torch.cat([torch.full((labels.shape[0], 1), start_token, device=device), labels[:, :-1]], dim=1)
                stage1_logits = stage1_model.model(input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask'), decoder_input_ids=decoder_input_ids).logits
                
                stage1_losses = torch.nn.functional.cross_entropy(stage1_logits.view(-1, stage1_logits.size(-1)), labels.view(-1), ignore_index=tokenizer.pad_token_id, reduction='none').view_as(labels)
                stage1_probs = torch.nn.functional.softmax(stage1_logits, dim=-1)
                stage1_entropies = -torch.sum(stage1_probs * torch.log(stage1_probs + 1e-9), dim=-1)

                valid_mask = labels != tokenizer.pad_token_id
                if valid_mask.sum() > 0:
                    all_ats_temperatures.extend(ats_temperatures[valid_mask].cpu().numpy().flatten())
                    all_stage1_losses.extend(stage1_losses[valid_mask].cpu().numpy().flatten())
                    all_stage1_entropies.extend(stage1_entropies[valid_mask].cpu().numpy().flatten())
            except Exception as e:
                print(f"Skipping a sample due to error: {e}")
                continue
    
    if not all_stage1_losses:
        return 0.0, 0.0

    ats_corr, _ = stats.pearsonr(all_ats_temperatures, all_stage1_losses)
    entropy_corr, _ = stats.pearsonr(all_stage1_entropies, all_stage1_losses)
    
    return ats_corr, entropy_corr

def plot_correlation_report(layer_counts, ats_correlations, entropy_baseline_corr, save_path):
    """生成并保存在不同层数下的相关性对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.5
    index = np.arange(len(layer_counts))
    
    # 绘制ATS相关性的条形图
    bars = ax.bar(index, ats_correlations, bar_width, label='ATS Temperature Correlation', color='skyblue')

    # 绘制词元熵相关性的基准线
    ax.axhline(y=entropy_baseline_corr, color='r', linestyle='--', linewidth=2, label=f'Token Entropy Baseline ({entropy_baseline_corr:.3f})')
    
    ax.set_xlabel('Number of Transformer Layers in ATS Head', fontsize=12)
    ax.set_ylabel('Pearson Correlation with Model Loss', fontsize=12)
    ax.set_title('ATS Head Performance vs. Token Entropy Baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(index)
    ax.set_xticklabels([f'{l} Layer(s)' for l in layer_counts])
    ax.set_ylim(bottom=0)
    ax.legend()
    
    # 在条形图上添加数值标签
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nCorrelation report chart saved to: {save_path}")

def main():
    print("Generating correlation report for different ATS Head layer counts...")
    
    # --- 配置 ---
    model_name = "Salesforce/codet5p-220m"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    stage1_model_path = "/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/logs_full_data_stable/stage1/final_model_with_new_temperature_stage1.ckpt"
    
    stage2_model_configs = {
        1: "/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/logs_full_data_stable/stage2/final_model_new_temperature_newloss_layer1_stage2.ckpt",
        2: "/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/logs_full_data_stable/stage2/final_model_with_new_tempurature_stage2_with_new_loss.ckpt", # Assuming this is 2 layers
        3: "/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/logs_full_data_stable/stage2/final_model_new_temperature_newloss_layer3_stage2.ckpt",
        4: "/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/logs_full_data_stable/stage2/final_model_new_temperature_newloss_layer4_stage2.ckpt",
    }
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 加载 ---
    print(f"Using device: {device}")
    tokenizer, _ = load_hf_tokenizer(model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)

    print(f"Loading Stage 1 model from: {stage1_model_path}")
    base_model_stage1 = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
    stage1_model = T5Module(base_model_stage1, hidden_size=base_model_stage1.config.d_model, pad_token=tokenizer.pad_token_id, train_mode="t5")
    stage1_model.load_state_dict(torch.load(stage1_model_path, map_location=device)["state_dict"], strict=False)
    stage1_model.to(device)

    # --- 分析 ---
    layer_counts = []
    ats_correlations = []
    entropy_baseline_corr = None

    for num_layers, stage2_path in sorted(stage2_model_configs.items()):
        if not os.path.exists(stage2_path):
            print(f"Warning: Stage 2 model for {num_layers} layers not found at {stage2_path}. Skipping.")
            continue
            
        print(f"\n--- Analyzing ATS Head with {num_layers} layer(s) ---")
        print(f"Loading Stage 2 model from: {stage2_path}")
        
        base_model_stage2 = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
        ats_model = T5Module(base_model_stage2, hidden_size=base_model_stage2.config.d_model, pad_token=tokenizer.pad_token_id, train_mode="ats", ats_head_layers=num_layers)
        ats_model.load_state_dict(torch.load(stage2_path, map_location=device)["state_dict"])
        ats_model.to(device)
        
        ats_corr, entropy_corr = get_correlation_scores(ats_model, stage1_model, tokenizer, device)
        
        print(f"Result for {num_layers} layer(s): ATS Correlation = {ats_corr:.4f}")
        
        layer_counts.append(num_layers)
        ats_correlations.append(ats_corr)
        
        # Entropy baseline is calculated each time but should be consistent. We take the last one.
        if entropy_corr is not None:
            entropy_baseline_corr = entropy_corr

    if not layer_counts:
        print("Error: No valid models were analyzed. Cannot generate report.")
        return

    print(f"\nFinal Token Entropy Baseline Correlation: {entropy_baseline_corr:.4f}")

    # --- 可视化 ---
    chart_path = os.path.join(output_dir, "ats_layer_vs_entropy_correlation_report.png")
    plot_correlation_report(layer_counts, ats_correlations, entropy_baseline_corr, chart_path)

if __name__ == "__main__":
    main() 