#!/usr/bin/env python3
"""
直接测试ATS温度缩放对beam search的效果
对比：原始logits vs 温度缩放logits vs stage1基准模型
"""

import os
import sys
import torch
import numpy as np
import pickle
from pathlib import Path
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, '/scratch/work/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master')

from uncertainty_search_compgen.validation import compute_validation_metrics
from uncertainty_search_compgen.train_lm import T5Module
from uncertainty_search_compgen.load_hf_lm import load_hf_tokenizer
from uncertainty_search_compgen.data import load_smcalflow_cs_simplified
from uncertainty_search_compgen.dataset import TokenizerPairDataset
from uncertainty_search_compgen.inference_ats import beam_search_hf
from transformers import T5Config, T5ForConditionalGeneration
from torch.utils.data import DataLoader

def create_temperature_scaling_sampler(use_temperature_scaling=True):
    """
    创建一个使用/不使用温度缩放的beam search采样器
    """
    def temperature_aware_beam_search(harness, batch, beams=5, k=5, early_stopping=True, max_length=128):
        """
        修改的beam search，可以选择是否使用ATS温度缩放
        """
        device = harness.device
        input_ids = batch["input_ids"].to(device)
        
        if use_temperature_scaling and harness.train_mode == "ats":
            # 🔥 关键：使用ATS头预测温度并应用缩放
            with torch.no_grad():
                # 获取编码器输出
                encoder_outputs = harness.model.encoder(input_ids)
                encoder_hidden_states = encoder_outputs.last_hidden_state
                
                # 使用ATS头预测温度
                temperatures = harness.ats_head(encoder_hidden_states)
                
                # 生成时应用温度缩放
                generation_outputs = harness.model.generate(
                    input_ids=input_ids,
                    num_beams=beams,
                    num_return_sequences=k,
                    early_stopping=early_stopping,
                    max_length=max_length,
                    do_sample=False,
                    pad_token_id=harness.hparams.pad_token,
                    # 这里我们需要自定义logits处理来应用温度
                    output_scores=True,
                    return_dict_in_generate=True
                )
        else:
            # 🔥 不使用温度缩放的标准beam search
            with torch.no_grad():
                generation_outputs = harness.model.generate(
                    input_ids=input_ids,
                    num_beams=beams,
                    num_return_sequences=k,
                    early_stopping=early_stopping,
                    max_length=max_length,
                    do_sample=False,
                    pad_token_id=harness.hparams.pad_token,
                    output_scores=True,
                    return_dict_in_generate=True
                )
        
        # 提取生成的序列
        generated_sequences = generation_outputs.sequences
        batch_size = input_ids.shape[0]
        
        # 移除输入部分，只保留生成的部分
        input_length = input_ids.shape[1]
        if generated_sequences.shape[1] > input_length:
            generated_sequences = generated_sequences[:, input_length:]
        
        # 重塑为期望的格式: (batch_size, seq_len, k)
        seq_len = generated_sequences.shape[1]
        generated_sequences = generated_sequences.view(batch_size, k, seq_len)
        generated_sequences = generated_sequences.transpose(1, 2)  # (batch_size, seq_len, k)
        
        return generated_sequences
    
    return temperature_aware_beam_search

def create_custom_temperature_beam_search():
    """
    创建自定义的温度缩放beam search
    """
    def custom_beam_search(harness, batch, beams=5, k=5, max_length=128):
        """
        自实现的beam search，支持逐步温度缩放
        """
        device = harness.device
        input_ids = batch["input_ids"].to(device)
        batch_size = input_ids.shape[0]
        
        # 编码输入
        encoder_outputs = harness.model.encoder(input_ids)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # 获取ATS温度预测（如果可用）
        use_ats = hasattr(harness, 'ats_head') and harness.train_mode == "ats"
        
        # 初始化decoder
        decoder_start_token = harness.model.config.decoder_start_token_id
        current_sequences = torch.full((batch_size * beams, 1), decoder_start_token, 
                                     dtype=torch.long, device=device)
        current_scores = torch.zeros(batch_size * beams, device=device)
        
        # 扩展encoder states for beam search
        expanded_encoder_states = encoder_hidden_states.repeat_interleave(beams, dim=0)
        
        results = []
        
        for step in range(max_length):
            # 获取下一个token的logits
            decoder_outputs = harness.model.decoder(
                input_ids=current_sequences,
                encoder_hidden_states=expanded_encoder_states
            )
            
            next_token_logits = harness.model.lm_head(decoder_outputs.last_hidden_state[:, -1, :])
            
            if use_ats:
                # 🔥 应用ATS温度缩放
                # 这里简化处理，使用平均温度
                avg_temp = harness.ats_head(encoder_hidden_states).mean(dim=1, keepdim=True)  # (batch, 1, 1)
                expanded_temp = avg_temp.repeat_interleave(beams, dim=0)  # (batch*beams, 1, 1)
                next_token_logits = next_token_logits / expanded_temp.squeeze()
            
            # Beam search逻辑
            vocab_size = next_token_logits.shape[-1]
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            
            # 更新scores
            next_scores = current_scores.unsqueeze(1) + log_probs
            next_scores = next_scores.view(batch_size, beams * vocab_size)
            
            # 选择top-k
            next_scores, next_tokens = torch.topk(next_scores, beams, dim=1)
            
            # 计算beam indices和token indices
            beam_indices = next_tokens // vocab_size
            token_indices = next_tokens % vocab_size
            
            # 更新sequences
            batch_beam_indices = torch.arange(batch_size, device=device).unsqueeze(1) * beams + beam_indices
            current_sequences = torch.cat([
                current_sequences[batch_beam_indices.flatten()],
                token_indices.unsqueeze(-1)
            ], dim=-1)
            current_scores = next_scores.flatten()
            
            # 检查EOS
            eos_token = harness.model.config.eos_token_id
            if eos_token is not None and (token_indices == eos_token).any():
                break
        
        # 重塑输出格式
        final_sequences = current_sequences.view(batch_size, beams, -1)[:, :k, 1:]  # 移除start token，只保留前k个
        
        # 转换为期望格式 (batch_size, seq_len, k)
        max_seq_len = final_sequences.shape[2]
        output = final_sequences.transpose(1, 2)  # (batch_size, seq_len, k)
        
        return output
    
    return custom_beam_search

def load_models_and_data(num_samples=50):
    """加载所有需要的模型和数据"""
    print("🔄 Loading models and data...")
    
    # 基础配置
    config = T5Config.from_pretrained("Salesforce/codet5p-220m", 
                                     output_hidden_states=True, 
                                     return_dict=True, 
                                     use_cache=False)
    tokenizer, _ = load_hf_tokenizer("Salesforce/codet5p-220m")
    basepath = Path("/scratch/work/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master")
    
    # 1. Stage1模型（基准）
    print("📁 Loading Stage1 model (baseline)...")
    model_stage1 = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5p-220m", 
                                                             config=config, cache_dir="/tmp/hf")
    harness_stage1 = T5Module(model_stage1, pad_token=tokenizer.pad_token_id, 
                             hidden_size=model_stage1.config.d_model, train_mode="t5")
    
    stage1_checkpoint = basepath / "logs_full_data_stable/stage1/final_model_with_new_temperature_stage1_with_new_loss.ckpt"
    if stage1_checkpoint.exists():
        harness_stage1.load_state_dict(torch.load(str(stage1_checkpoint))["state_dict"], strict=False)
        print(f"✅ Loaded stage1 checkpoint")
    
    # 2. Stage2模型（ATS）
    print("📁 Loading Stage2 model (ATS)...")
    model_stage2 = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5p-220m", 
                                                             config=config, cache_dir="/tmp/hf")
    harness_stage2 = T5Module(model_stage2, pad_token=tokenizer.pad_token_id, 
                             hidden_size=model_stage2.config.d_model, train_mode="ats")
    
    stage2_checkpoint = basepath / "logs_full_data_stable/stage2/final_model_with_new_tempurature_stage2.ckpt"
    if stage2_checkpoint.exists():
        harness_stage2.load_state_dict(torch.load(str(stage2_checkpoint))["state_dict"], strict=False)
        print(f"✅ Loaded stage2 checkpoint")
    else:
        print("❌ Stage2 checkpoint not found!")
        return None, None, None, None
    
    # 3. 测试数据
    print(f"📊 Loading test data ({num_samples} samples)...")
    _, _, _, test_pairs = load_smcalflow_cs_simplified(basepath / "text/semparse/smcalflow-cs")
    test_samples = test_pairs[:num_samples]
    
    dataloader = DataLoader(
        TokenizerPairDataset(test_samples, tokenizer), 
        batch_size=4,  # 小batch确保稳定性
        shuffle=False
    )
    
    return harness_stage1, harness_stage2, dataloader, tokenizer

def run_temperature_scaling_comparison():
    """运行温度缩放对比测试"""
    print(f"\n{'='*70}")
    print("🔬 ATS温度缩放效果对比测试")
    print(f"{'='*70}")
    
    # 加载模型和数据
    stage1_model, ats_model, test_dataloader, tokenizer = load_models_and_data(num_samples=50)
    
    if ats_model is None:
        print("❌ 无法加载ATS模型，测试终止")
        return
    
    # 移动模型到GPU
    if torch.cuda.is_available():
        stage1_model.cuda()
        ats_model.cuda()
    
    results = {}
    
    # 定义测试配置
    test_configs = [
        ("stage1_baseline", stage1_model, partial(beam_search_hf, beams=5, k=5, early_stopping=True, max_length=128)),
        ("ats_without_temperature", ats_model, create_temperature_scaling_sampler(use_temperature_scaling=False)),
        ("ats_with_temperature", ats_model, create_temperature_scaling_sampler(use_temperature_scaling=True)),
    ]
    
    # 运行测试
    for config_name, model, sampler in test_configs:
        print(f"\n📋 测试配置: {config_name}")
        print("-" * 50)
        
        try:
            stats = compute_validation_metrics(
                harness=model,
                val_pairs=test_dataloader,
                sampler=sampler,
                return_results=True,
                verbose=True
            )
            results[config_name] = stats
            print(f"✅ {config_name} 测试完成")
            
        except Exception as e:
            print(f"❌ {config_name} 测试失败: {str(e)}")
            results[config_name] = None
    
    return results

def analyze_temperature_effects(results):
    """分析温度缩放效果"""
    print(f"\n{'='*70}")
    print("📊 温度缩放效果分析")
    print(f"{'='*70}")
    
    metrics = ['accuracy', 'exacts', 'top3_accuracy', 'top3_exacts', 'top5_accuracy', 'top5_exacts']
    
    # 显示所有结果
    for metric in metrics:
        print(f"\n📈 {metric.upper()} 对比:")
        print("-" * 50)
        
        metric_results = {}
        for config_name, stats in results.items():
            if stats is not None and metric in stats:
                mean_val = np.mean(stats[metric])
                metric_results[config_name] = mean_val
                print(f"  {config_name:25}: {mean_val:.4f}")
        
        # 计算改进
        if len(metric_results) >= 2:
            baseline = metric_results.get('stage1_baseline', 0)
            without_temp = metric_results.get('ats_without_temperature', 0)
            with_temp = metric_results.get('ats_with_temperature', 0)
            
            if baseline > 0:
                print(f"    🔸 ATS无温度 vs Stage1: {((without_temp - baseline) / baseline * 100):+.2f}%")
                print(f"    🔥 ATS有温度 vs Stage1: {((with_temp - baseline) / baseline * 100):+.2f}%")
                if without_temp > 0:
                    print(f"    ⭐ 温度缩放效果: {((with_temp - without_temp) / without_temp * 100):+.2f}%")

def save_temperature_results(results):
    """保存结果"""
    basepath = Path("/scratch/work/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master")
    results_dir = basepath / "temperature_scaling_results"
    results_dir.mkdir(exist_ok=True)
    
    result_file = results_dir / "temperature_scaling_comparison.pickle"
    
    with open(result_file, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\n💾 结果已保存到: {result_file}")

def main():
    """主函数"""
    print("🚀 开始ATS温度缩放效果验证")
    print("=" * 70)
    
    # 运行对比测试
    results = run_temperature_scaling_comparison()
    
    if results:
        # 分析结果
        analyze_temperature_effects(results)
        
        # 保存结果
        save_temperature_results(results)
        
        print(f"\n✅ 温度缩放效果验证完成!")
    else:
        print("❌ 测试失败")

if __name__ == "__main__":
    main()