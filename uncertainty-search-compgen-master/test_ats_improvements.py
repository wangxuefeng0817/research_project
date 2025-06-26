#!/usr/bin/env python3
"""
测试ATS温度、词元熵与Stage1损失相关性的脚本
1. 比较ATS模型预测的温度与Stage1模型计算的损失是否高度相关
2. 比较Stage1模型计算的词元熵与Stage1模型计算的损失是否高度相关
"""
from uncertainty_search_compgen.load_hf_lm import load_hf_tokenizer 
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig
from uncertainty_search_compgen.train_lm import T5Module, ATShead
from uncertainty_search_compgen.dataset import TokenizerPairIterableDataset
import matplotlib.pyplot as plt
import json
from scipy import stats
import warnings
import os
import torch
from uncertainty_search_compgen.data import load_smcalflow_cs_simplified
import argparse
warnings.filterwarnings('ignore')

def test_uncertainty_correlation(ats_model, stage1_model, tokenizer, test_samples, device="cuda"):
    """测试不确定性度量（ATS温度、词元熵）与Stage1损失的相关性"""
    ats_model.eval()
    stage1_model.eval()
    
    all_ats_temperatures = []
    all_stage1_losses = []
    all_stage1_entropies = []
    all_tokens = []
    
    with torch.no_grad():
        for i, sample in enumerate(test_samples[:100]):  
            if i % 10 == 0:
                print(f"处理样本 {i}/100...")
                
            # 准备输入
            inputs = tokenizer(
                sample['input'], 
                max_length=256, 
                truncation=True, 
                padding='max_length',
                return_tensors='pt'
            )
            labels = tokenizer(
                sample['output'],
                max_length=256,
                truncation=True,
                padding='max_length', 
                return_tensors='pt'
            )['input_ids']
            
            # 移动到设备
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            inputs['labels'] = labels
            
            # 🔥 Step 1: 用ATS模型预测温度
            try:
                scaled_logits, ats_temperatures, original_logits = ats_model(inputs)
                
                # 🔥 Step 2: 用Stage1模型计算损失和熵
                decoder_input_ids = labels.clone()
                if hasattr(stage1_model.model.config, 'decoder_start_token_id') and stage1_model.model.config.decoder_start_token_id is not None:
                    start_token = stage1_model.model.config.decoder_start_token_id
                else:
                    start_token = tokenizer.pad_token_id
                
                batch_size, seq_len = labels.shape
                decoder_input_ids = torch.cat([
                    torch.full((batch_size, 1), start_token, device=labels.device),
                    labels[:, :-1]
                ], dim=1)
                
                stage1_model_inputs = {
                    'input_ids': inputs['input_ids'],
                    'decoder_input_ids': decoder_input_ids,
                    'attention_mask': inputs.get('attention_mask'),
                }
                
                stage1_logits = stage1_model.model(**stage1_model_inputs).logits
                
                # 计算Stage1模型的交叉熵损失
                stage1_losses = torch.nn.functional.cross_entropy(
                    stage1_logits.view(-1, stage1_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=tokenizer.pad_token_id,
                    reduction='none'
                ).view_as(labels)
                
                # 🔥 新增：计算Stage1模型的词元熵
                stage1_probs = torch.nn.functional.softmax(stage1_logits, dim=-1)
                stage1_entropies = -torch.sum(stage1_probs * torch.log(stage1_probs + 1e-9), dim=-1)

                # 提取有效token（非padding）
                valid_mask = labels != tokenizer.pad_token_id
                if valid_mask.sum() > 0:
                    # ATS温度
                    if hasattr(ats_temperatures, 'mean') and ats_temperatures.numel() > 1:
                        valid_ats_temps = ats_temperatures[valid_mask].cpu().numpy().flatten()
                    else:
                        continue
                    
                    valid_stage1_losses = stage1_losses[valid_mask].cpu().numpy().flatten()
                    valid_stage1_entropies = stage1_entropies[valid_mask].cpu().numpy().flatten()
                    valid_tokens = labels[valid_mask].cpu().numpy().flatten()
                    
                    all_ats_temperatures.extend(valid_ats_temps)
                    all_stage1_losses.extend(valid_stage1_losses)
                    all_stage1_entropies.extend(valid_stage1_entropies)
                    all_tokens.extend(valid_tokens)
            
            except Exception as e:
                print(f"处理样本{i}时出错: {e}")
                continue
    
    return (
        np.array(all_ats_temperatures), 
        np.array(all_stage1_losses), 
        np.array(all_stage1_entropies),
        np.array(all_tokens)
    )

def analyze_correlation(signal_values, loss_values, signal_name):
    """分析不确定性信号与损失的交叉相关性"""
    print(f"--- {signal_name} vs Stage1 Loss Correlation Analysis ---\n")
    
    if len(signal_values) == 0:
        print("No valid data to analyze.")
        return {}

    # 1. 基本统计
    print("1. Basic Statistics:")
    print(f"   Sample Count: {len(signal_values)}")
    print(f"   {signal_name} Range: [{signal_values.min():.4f}, {signal_values.max():.4f}]")
    print(f"   {signal_name} Mean: {signal_values.mean():.4f}")
    print(f"   Stage1 Loss Range: [{loss_values.min():.4f}, {loss_values.max():.4f}]")
    print(f"   Stage1 Loss Mean: {loss_values.mean():.4f}")
    
    # 2. 相关性分析
    correlation, p_value = stats.pearsonr(signal_values, loss_values)
    spearman_corr, spearman_p = stats.spearmanr(signal_values, loss_values)
    
    print(f"\n2. Correlation Analysis:")
    print(f"   Pearson Correlation: {correlation:.4f} (p={p_value:.6f})")
    print(f"   Spearman Correlation: {spearman_corr:.4f} (p={spearman_p:.6f})")
    
    if abs(correlation) > 0.7:
        strength = "Very Strong"
        interpretation = f"The {signal_name} is an excellent indicator of model uncertainty!"
    elif abs(correlation) > 0.5:
        strength = "Strong"
        interpretation = f"The {signal_name} is a good indicator of model uncertainty."
    elif abs(correlation) > 0.3:
        strength = "Moderate"
        interpretation = f"The {signal_name} captures some uncertainty, but there's room for improvement."
    else:
        strength = "Weak"
        interpretation = f"The {signal_name} may not be effectively capturing the uncertainty signal."
    
    print(f"   Correlation Strength: {strength}")
    print(f"   Interpretation: {interpretation}")
    
    # 3. 分位数分析
    loss_q25, loss_q75 = np.percentile(loss_values, [25, 75])
    high_loss_mask = loss_values > loss_q75
    low_loss_mask = loss_values < loss_q25
    
    high_loss_signal = signal_values[high_loss_mask]
    low_loss_signal = signal_values[low_loss_mask]
    
    print(f"\n3. Quantile Analysis:")
    print(f"   Avg. {signal_name} for High-Loss Tokens (>75%): {high_loss_signal.mean():.4f}")
    print(f"   Avg. {signal_name} for Low-Loss Tokens (<25%): {low_loss_signal.mean():.4f}")
    
    signal_diff = high_loss_signal.mean() - low_loss_signal.mean()
    print(f"   Signal Difference: {signal_diff:.4f}")

    return {
        'pearson_correlation': correlation,
        'pearson_p_value': p_value,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'signal_mean': signal_values.mean(),
        'signal_std': signal_values.std(),
        'loss_mean': loss_values.mean(),
        'loss_std': loss_values.std(),
        'high_loss_signal_mean': high_loss_signal.mean(),
        'low_loss_signal_mean': low_loss_signal.mean(),
        'signal_diff': signal_diff,
        'strength': strength,
        'interpretation': interpretation
    }

def create_correlation_visualization(signal_values, loss_values, signal_name, save_path_prefix):
    """
    创建不确定性信号 vs 损失的可视化图表，并分别保存。
    """
    # --- 图一：散点图 ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(signal_values, loss_values, alpha=0.5, s=5, c='blue')
    ax1.set_xlabel(f'{signal_name}', fontsize=12)
    ax1.set_ylabel('Stage1 Model Loss', fontsize=12)
    ax1.set_title(f'{signal_name} vs. Stage1 Loss', fontsize=14, fontweight='bold')
    
    # 添加趋势线
    z = np.polyfit(signal_values, loss_values, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(signal_values.min(), signal_values.max(), 100)
    ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    
    # 添加相关系数
    correlation, _ = stats.pearsonr(signal_values, loss_values)
    ax1.text(0.05, 0.95, f'Pearson r = {correlation:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle="round", facecolor='white', alpha=0.8), fontsize=12)
    
    plt.tight_layout()
    scatter_path = f"{save_path_prefix}_scatter.png"
    plt.savefig(scatter_path, dpi=300)
    print(f"\nScatter plot saved to: {scatter_path}")
    plt.close(fig1)

    # --- 图二：条形图 ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    loss_quartiles = np.percentile(loss_values, [0, 25, 50, 75, 100])
    signal_by_loss_quartile = []
    quartile_labels = ['Q1 (Low Loss)', 'Q2', 'Q3', 'Q4 (High Loss)']
    
    for i in range(4):
        mask = (loss_values >= loss_quartiles[i]) & (loss_values < loss_quartiles[i+1])
        if mask.sum() > 0:
            signal_by_loss_quartile.append(signal_values[mask].mean())
        else:
            signal_by_loss_quartile.append(0)
    
    bars = ax2.bar(quartile_labels, signal_by_loss_quartile, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_ylabel(f'Average {signal_name}', fontsize=12)
    ax2.set_title(f'Average {signal_name} by Stage1 Loss Quartile', fontsize=14, fontweight='bold')
    
    # 添加数值标签
    for bar, value in zip(bars, signal_by_loss_quartile):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    barchart_path = f"{save_path_prefix}_barchart.png"
    plt.savefig(barchart_path, dpi=300)
    print(f"Bar chart saved to: {barchart_path}")
    plt.close(fig2)

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
    parser = argparse.ArgumentParser(description="分析不确定性度量与模型损失的相关性")
    parser.add_argument("--model_name", type=str, default="Salesforce/codet5p-220m", help="基础模型名称")
    parser.add_argument("--stage1_path", type=str, help="Stage-1模型的路径", default=None)
    parser.add_argument("--dataset", type=str, choices=['virtual', 'smcalflow'], default='smcalflow', help="选择使用的数据集")
    parser.add_argument("--num_samples", type=int, default=100, help="测试样本数量")
    args = parser.parse_args()

    print(f"--- 开始生成ATS头层数与不确定性度量的对比报告 ---")
    print(f"--- 数据集: '{args.dataset}', 样本数: {args.num_samples} ---")

    # --- 配置 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- 加载通用组件 ---
    tokenizer, _ = load_hf_tokenizer(args.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model_name, output_hidden_states=True)

    # --- 数据加载 ---
    if args.dataset == 'virtual':
        print("\nPreparing virtual test data...")
        test_data = [
            {"input": "What is 2 + 3?", "output": "5"},
            {"input": "What is the capital of France?", "output": "Paris"},
            {"input": "Translate Hello to French", "output": "Bonjour"},
            {"input": "Write a function to add two numbers", "output": "def add(a, b): return a + b"},
        ] * (args.num_samples // 4)
        print(f"Using {len(test_data)} virtual test samples.")
    else: # smcalflow
        print("\nPreparing SMCalFlow test data...")
        data_path = "/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/text/semparse/smcalflow-cs"
        try:
            _, _, test_pairs, _ = load_smcalflow_cs_simplified(data_path)
            test_data = [{"input": src, "output": tgt} for src, tgt, qid in test_pairs[:args.num_samples]]
            print(f"Successfully loaded {len(test_data)} SMCalFlow test samples.")
        except Exception as e:
            print(f"Error loading SMCalFlow data: {e}")
            return
            
    # --- 模型路径配置 ---
    stage1_model_path = args.stage1_path or "/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/logs_full_data_stable/stage1/final_model_with_new_temperature_stage1.ckpt"
    stage2_model_configs = {
        1: "/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/logs_full_data_stable/stage2/final_model_new_temperature_newloss_layer1_stage2.ckpt",
        2: "/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/logs_full_data_stable/stage2/final_model_with_new_tempurature_stage2_with_new_loss.ckpt",
        3: "/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/logs_full_data_stable/stage2/final_model_new_temperature_newloss_layer3_stage2.ckpt",
        4: "/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/logs_full_data_stable/stage2/final_model_new_temperature_newloss_layer4_stage2.ckpt",
    }
    
    # --- 加载基准Stage-1模型 ---
    print(f"\nLoading Stage 1 model from: {stage1_model_path}")
    base_model_stage1 = T5ForConditionalGeneration.from_pretrained(args.model_name, config=config)
    stage1_model = T5Module(base_model_stage1, hidden_size=base_model_stage1.config.d_model, pad_token=tokenizer.pad_token_id, train_mode="t5")
    stage1_model.load_state_dict(torch.load(stage1_model_path, map_location=device)["state_dict"], strict=False)
    stage1_model.to(device)
    print("✅ Stage 1 model loaded.")

    # --- 循环分析 ---
    layer_counts = []
    ats_correlations = []
    entropy_baseline_corr = None
    all_results = {}

    for num_layers, stage2_path in sorted(stage2_model_configs.items()):
        if not os.path.exists(stage2_path):
            print(f"\nWarning: Stage 2 model for {num_layers} layers not found at {stage2_path}. Skipping.")
            continue
            
        print(f"\n--- Analyzing ATS Head with {num_layers} layer(s) ---")
        print(f"Loading Stage 2 model from: {stage2_path}")
        
        base_model_stage2 = T5ForConditionalGeneration.from_pretrained(args.model_name, config=config)
        ats_model = T5Module(
            base_model_stage2, 
            hidden_size=base_model_stage2.config.d_model, 
            pad_token=tokenizer.pad_token_id, 
            train_mode="ats",
            ats_head_layers=num_layers
        )
        ats_model.load_state_dict(torch.load(stage2_path, map_location=device)["state_dict"])
        ats_model.to(device)
        print(f"✅ Stage 2 model with {num_layers} layer(s) loaded.")
        
        print("Running correlation analysis...")
        ats_temps, losses, entropies, _ = test_uncertainty_correlation(
            ats_model, stage1_model, tokenizer, test_data, device)
        
        if len(ats_temps) == 0:
            print("Error: No valid data collected for this model.")
            continue
        
        ats_results = analyze_correlation(ats_temps, losses, f"ATS Temperature ({num_layers}-layer)")
        layer_counts.append(num_layers)
        ats_correlations.append(ats_results.get('pearson_correlation', 0.0))
        
        all_results[f'ats_{num_layers}_layer'] = ats_results

        if entropy_baseline_corr is None:
            entropy_results = analyze_correlation(entropies, losses, "Token Entropy")
            entropy_baseline_corr = entropy_results.get('pearson_correlation', 0.0)
            all_results['token_entropy'] = entropy_results
            print(f"Established Token Entropy Baseline Correlation: {entropy_baseline_corr:.4f}")

    # --- 可视化与保存 ---
    if not layer_counts:
        print("\nError: No valid models were analyzed. Cannot generate report.")
        return

    output_dir = f"figures_{args.dataset}_{args.num_samples}samples"
    os.makedirs(output_dir, exist_ok=True)
    
    chart_path = os.path.join(output_dir, "ats_layer_vs_entropy_correlation_report.png")
    plot_correlation_report(layer_counts, ats_correlations, entropy_baseline_corr, chart_path)

    # 保存所有分析结果
    json_path = os.path.join(output_dir, "full_correlation_results.json")
    def convert_to_python_type(obj):
        if isinstance(obj, np.generic): return obj.item()
        if isinstance(obj, dict): return {k: convert_to_python_type(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert_to_python_type(v) for v in obj]
        return obj
    final_data = convert_to_python_type(all_results)
    
    with open(json_path, "w") as f:
        json.dump(final_data, f, indent=2)
    
    print(f"\nDetailed analysis saved to: {json_path}")
    print("\n=== Report Generation Complete ===")

if __name__ == "__main__":
    main() 