
import torch
import os
import sys
sys.path.append("/home/wangx36/.local/lib/python3.11/site-packages")
from uncertainty_search_compgen.data import load_smcalflow_cs_simplified
from uncertainty_search_compgen.load_hf_lm import load_hf_tokenizer
from uncertainty_search_compgen.train_lm import finetune_tf_lm,T5Module,compute_loss_and_accuracy_first
from uncertainty_search_compgen.inference import get_topk_outputs, generate_gpt, uncertainty_guided_search
from uncertainty_search_compgen.dataset import TokenizerPairDataset, TokenizerPairIterableDataset,Tokenizer_with_attention_mask
from uncertainty_search_compgen.divergence_metrics import (
    measure_entropy,
    measure_mutual_kl_causal_mask,
    measure_teacher_student_model_divergence
)
from uncertainty_search_compgen.plotting import visualize_as_table, plot_batch_and_metric, plot_batch_and_metric_pair
from torch.nn import CrossEntropyLoss
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from uncertainty_search_compgen.data import format_source, retokenize_input,parse_json_objects
print("Torch file path:", torch.__file__)
os.environ["HF_HOME"] = "/tmp/hf"
print(os.getcwd())
print(sys.executable)
(
    train_pairs,
    train_full,
    val_pairs,
    test_pairs
) = load_smcalflow_cs_simplified(
    "/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master/text/semparse/smcalflow-cs"
)
tokenizer, idx2word = load_hf_tokenizer("Salesforce/codet5p-220m")
vocab_size = tokenizer.vocab_size
print(f"tokenizer_size: {vocab_size}")
train_only_trainer, train_only_model1, train_only_harness = finetune_tf_lm(
    "Salesforce/codet5p-220m",
    tokenizer,
    train_pairs[:-128],
    train_full,
    val_pairs,
    load_from="logs/train_only/stage1/final_model_with_new_temperature_stage1.ckpt",  # 加载training only的模型
    do_fit=False  # 不重新训练
)

# model_vocab_size = train_only_model1.get_output_embeddings().weight.shape[0]
# print(f"Model effective vocab size: {model_vocab_size}")

data_for_calculate_loss = Tokenizer_with_attention_mask(list(test_pairs), tokenizer)

# data_iter = iter(data_for_calculate_loss)
# sample = next(data_iter)
loss_fn_per_token = torch.nn.CrossEntropyLoss(
    reduction='none',                        # <--- 关键：告诉它不要聚合损失，返回每个元素的损失
    ignore_index= -100 # <--- 关键：告诉它忽略填充标记
)

# print(sample["input_ids"])
# print(data_for_calculate_loss[0]['input_ids'])
import torch

# 如果有GPU则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_data = DataLoader(data_for_calculate_loss, batch_size=52)
train_only_harness.to(device)
train_only_harness.eval()
all_token_losses = []
all_labels_ref = []
with torch.no_grad():   
    for i, batch in enumerate(test_data):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = batch["labels"].to(device)
        device_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        scaled_logits, temperatures, logits = train_only_harness(device_batch)
        
        batch_size, target_seq_len, vocab_size = logits.shape
        print(logits.shape)
        
        # Logits: (N, C) -> (batch_size * target_seq_len, vocab_size)
        # Labels: (N) -> (batch_size * target_seq_len)
        reshaped_logits = logits.view(-1, vocab_size)
        # print(f"reshaped_logits.shape:{reshaped_logits.shape}")
        reshaped_labels = labels.view(-1)
        # print(f"reshaped_labels.shape:{reshaped_labels.shape}")
        # min_val = reshaped_labels.min().item()
        # max_val = reshaped_labels.max().item()
        # model_vocab_size = reshaped_logits.shape[-1] # Should be 32100
        # print(f"Batch {i+1}: Labels min={min_val}, max={max_val}. Model vocab size={model_vocab_size}")

        # (batch_size * target_seq_len)
        token_losses = loss_fn_per_token(reshaped_logits, reshaped_labels)

        # (batch_size, target_seq_len)
        token_losses = token_losses.view(batch_size, target_seq_len)

   
        all_token_losses.append(token_losses.cpu())
        all_labels_ref.append(labels.cpu()) # save labels


all_token_losses_tensor = torch.cat(all_token_losses, dim=0)
all_labels_ref_tensor = torch.cat(all_labels_ref, dim=0)

print(f"Shape of per-token losses tensor: {all_token_losses_tensor.shape}")
print("\nLosses for the first sample (non-padding tokens):")
first_sample_losses = all_token_losses_tensor[0]
first_sample_labels = all_labels_ref_tensor[0]
for i, (loss, label_id) in enumerate(zip(first_sample_losses, first_sample_labels)):
    if label_id != -100: 
        token = tokenizer.decode(label_id.item())
        print(f"Token {i} ('{token}'): Loss = {loss.item():.4f}")
        

results = []

for sample_idx in range(len(all_token_losses_tensor)):
    sample_losses = all_token_losses_tensor[sample_idx]
    sample_labels = all_labels_ref_tensor[sample_idx]
    
    
    for token_idx, (loss, label_id) in enumerate(zip(sample_losses, sample_labels)):
        if label_id != -100:  # 只保存非填充标记
            token = tokenizer.decode(label_id.item())
            results.append({
                'token_idx': token_idx,
                'token': token,
                'loss': loss.item()
            })


results.sort(key = lambda x:x["loss"], reverse=True)

    
# 保存为JSON
import json
with open('token_losses_readable_sorted_with_default_temperature_model.json', 'w') as f:
    json.dump(results, f, indent=2)