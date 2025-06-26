import sys
import os
import gc

project_root = "/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master"
if project_root not in sys.path:
    sys.path.append(project_root)


import os
os.chdir("/home/wangx36/uncertainty-search-compgen-master/uncertainty-search-compgen-master")


import sys
sys.path.append("/home/wangx36/.local/lib/python3.11/site-packages")


if __name__ == "__main__":
    __package__ = "uncertainty_search_compgen"


from .scheduler import transformer_optimizer_config


import transformers
from torch.utils.data import IterableDataset, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from tqdm.auto import tqdm, trange
import numpy as np
import os, torch, platform, warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from positional_encodings.torch_encodings import PositionalEncoding1D
import pytorch_lightning as pl
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config,AutoConfig
from .scheduler import transformer_optimizer_config
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
import torchmetrics
import json
import pandas as pd
import itertools
import re
import pdb


from .dataset import TokenizerPairIterableDataset

def compute_loss_and_accuracy_first(preds,targets, pad_target_idx):
    actions_mask = targets == pad_target_idx

    loss = F.cross_entropy(
        preds.flatten(0, -2), targets.flatten().long(), ignore_index=pad_target_idx
    )

    argmax_preds = preds.argmax(dim=-1)

    truncated_matches = [
        (ap[~mask] == tgt[~mask])
        for ap, mask, tgt in zip(argmax_preds, actions_mask, targets)
    ]
    acc = torch.cat(truncated_matches).float().mean()
    exacts = torch.tensor([tm.all() for tm in truncated_matches]).float().mean()

    return {
        "loss": loss,
        "acc": acc,
        "exacts": exacts,
    }

# +
def compute_loss_and_accuracy(preds, original_logits, targets, pad_target_idx, alpha=0.5):
    argmax_preds = original_logits.argmax(dim=-1)
    actions_mask = targets == pad_target_idx
    
  
    ce_loss = F.cross_entropy(
        preds.flatten(0, -2), targets.flatten().long(), 
        ignore_index=pad_target_idx,
        reduction='none'
    ).view_as(targets)
    
 
    vocab_size = preds.size(-1)
    uniform_loss = -F.log_softmax(preds, dim=-1).mean(dim=-1)
    # uniform_loss = torch.special.entr(torch.softmax(preds, dim=-1) + 0.000001).sum(dim=-1)
    correct_mask = (argmax_preds == targets) & ~actions_mask
    
#     pdb.set_trace()

    loss = torch.where(
        correct_mask,
        (1 - alpha) * ce_loss,  
        alpha * uniform_loss     
    )
    loss = loss.masked_fill(actions_mask, 0.0)  
    loss = loss.sum() / (~actions_mask).sum()  
    
   
    truncated_matches = [
        (ap[~mask] == tgt[~mask])
        for ap, mask, tgt in zip(argmax_preds, actions_mask, targets)
    ]
    acc = torch.cat(truncated_matches).float().mean()
    exacts = torch.tensor([tm.all() for tm in truncated_matches]).float().mean()

    return {
        "loss": loss,
        "acc": acc,
        "exacts": exacts,
    }


# -

def compute_loss_and_accuracy_ats(scaled_logits, original_logits, temperatures, targets, pad_target_idx, alpha=0.5, temp_weight=1.5):
    """
    ATS专用损失函数，方案2.5：简化并稳定
    
    Args:
        temp_weight: 温度正则化权重（调整到1.5）
    """
    argmax_preds = original_logits.argmax(dim=-1)
    actions_mask = targets == pad_target_idx
    
    ce_loss = F.cross_entropy(
        scaled_logits.flatten(0, -2), targets.flatten().long(), 
        ignore_index=pad_target_idx,
        reduction='none'
    ).view_as(targets)
    
    vocab_size = scaled_logits.size(-1)
    uniform_loss = -F.log_softmax(scaled_logits, dim=-1).mean(dim=-1)
    
    original_ce_loss = F.cross_entropy(
        original_logits.flatten(0, -2), targets.flatten().long(), 
        ignore_index=pad_target_idx,
        reduction='none'
    ).view_as(targets)
    
    # 方案2.5：简化的温度正则化 (MSE对齐 + 温度分离)
    valid_mask = ~actions_mask
    temp_reg_loss = torch.tensor(0.0, device=temperatures.device)
    
    if valid_mask.sum() > 0:
        valid_original_loss = original_ce_loss[valid_mask]
        valid_temperatures = temperatures.squeeze(-1)[valid_mask]
        
        # 1. MSE对齐损失 (鼓励线性相关)
        normalized_loss = (valid_original_loss - valid_original_loss.min()) / (valid_original_loss.max() - valid_original_loss.min() + 1e-8)
        target_temp = 0.5 + normalized_loss * 1.5  # 映射到[0.5, 2.0]
        mse_alignment_loss = F.mse_loss(valid_temperatures, target_temp)
        
        # 2. 温度分离损失 (鼓励高低损失区域温度有差异)
        loss_q75 = torch.quantile(valid_original_loss, 0.75)
        loss_q25 = torch.quantile(valid_original_loss, 0.25)
        
        high_loss_mask = valid_original_loss > loss_q75
        low_loss_mask = valid_original_loss < loss_q25
        
        if high_loss_mask.sum() > 0 and low_loss_mask.sum() > 0:
            high_loss_temps = valid_temperatures[high_loss_mask]
            low_loss_temps = valid_temperatures[low_loss_mask]
            # 鼓励高低损失token的平均温度差异至少为0.3
            temp_separation_loss = F.relu(0.3 - (high_loss_temps.mean() - low_loss_temps.mean())) 
        else:
            temp_separation_loss = torch.tensor(0.0, device=temperatures.device)
            
        temp_reg_loss = mse_alignment_loss + temp_separation_loss
    
    correct_mask = (argmax_preds == targets) & ~actions_mask
    main_loss = torch.where(
        correct_mask,
        (1 - alpha) * ce_loss,
        alpha * uniform_loss
    )
    main_loss = main_loss.masked_fill(actions_mask, 0.0).sum() / valid_mask.sum().clamp(min=1)
    
    total_loss = main_loss + temp_weight * temp_reg_loss
    
    truncated_matches = [
        (ap[~mask] == tgt[~mask])
        for ap, mask, tgt in zip(argmax_preds, actions_mask, targets)
    ]
    acc = torch.cat(truncated_matches).float().mean()
    exacts = torch.tensor([tm.all() for tm in truncated_matches]).float().mean()

    return {
        "loss": total_loss,
        "ce_loss": main_loss,
        "temp_reg_loss": temp_reg_loss,
        "acc": acc,
        "exacts": exacts,
    }


def linear_with_warmup_schedule(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    min_lr_scale,
    last_epoch=-1,
    no_lr_decay=False,
):
    min_lr_logscale = min_lr_scale

    def lr_lambda(current_step):
        # Scale from 0 to 1
        if current_step <= num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        if no_lr_decay:
            return 1

        # Scale from 1 to min_lr_scale logarithmically
        #
        # So for example, if min_lr_logscale is -3, then
        # scale goes from 0 to -3 meaning that the lr multiplier
        # goes from 1, to 1e-1 at -1, to 1e-2 at -2 to 1e-3 at -3.
        scale = min(
            1,
            float(current_step - num_warmup_steps)
            / float(num_training_steps - num_warmup_steps),
        )
        logscale = scale * min_lr_logscale
        multiplier = 10**logscale

        return multiplier

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_parameter_names(model, exclude_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, exclude_layer_types)
            if not isinstance(child, tuple(exclude_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def transformer_optimizer_config(
    harness,
    lr,
    warmup_proportion=0.14,
    decay_power=-2,
    weight_decay=0,
    no_lr_decay=False,
    optimizer_func=None,
    optimizer_kwargs=None,
    train_mode="t5"
):
    decay_parameters = get_parameter_names(harness.trainer.model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    if train_mode == "t5":
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in harness.trainer.model.model.decoder.block.named_parameters()
                    if n in decay_parameters
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in harness.trainer.model.model.decoder.block.named_parameters()
                    if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]
    else:
        # 方案2.5：简化并稳定 - 同时训练主模型和ATS头，调整学习率
        optimizer_grouped_parameters = [
            # 主模型参数，使用正常学习率的10%
            {
                "params": [
                    p
                    for n, p in harness.trainer.model.model.decoder.block.named_parameters()
                    if n in decay_parameters
                ],
                "lr": lr * 0.1,  # 主模型使用0.1x学习率
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in harness.trainer.model.model.decoder.block.named_parameters()
                    if n not in decay_parameters
                ],
                "lr": lr * 0.1,
                "weight_decay": 0.0,
            },
            # ATS头参数，使用正常学习率
            {"params": harness.ats_head.parameters(), "lr": lr, "weight_decay": 0.0}  # ATS头使用1.0x学习率
        ]
    optimizer = (optimizer_func or optim.AdamW)(
        optimizer_grouped_parameters, lr=lr, **(optimizer_kwargs or {})
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": linear_with_warmup_schedule(
                optimizer,
                harness.trainer.max_steps * warmup_proportion,
                harness.trainer.max_steps,
                decay_power,
                no_lr_decay=no_lr_decay,
            ),
            "interval": "step",
            "frequency": 1,
        },
    }



class ATShead(nn.Module):
    def __init__(self, hidden_size, num_layers=4): # 回到3层Transformer
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=16, #
                dim_feedforward=4*hidden_size, # 恢复4x FFN
                dropout=0.1,
                batch_first=True,
                activation='gelu',
                norm_first=True
            ) for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_size) 
        
        # 简化温度预测网络 (2层MLP)
        self.temperature_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(), # 保留GELU
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        with torch.no_grad():
            self.temperature_proj[-1].weight.normal_(0, 0.02)
            self.temperature_proj[-1].bias.fill_(0.0)
    
    def generate_mask(self, sz):
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        return mask
        
    def forward(self, hidden_states):
        x = hidden_states.requires_grad_()
        seq_len = hidden_states.size(1)
        att_mask = self.generate_mask(seq_len).to(hidden_states.device)
    
        for layer in self.layers:
            x = layer(x, src_mask=att_mask)
        
        x = self.layer_norm(x) # 应用LayerNorm
        
        temp_logits = self.temperature_proj(x)
        temperatures = torch.sigmoid(temp_logits) * 1.5 + 0.5
        
        return temperatures


# +
# def check_gradient_flow(model, tokenizer):

#     print("...")


#     test_input = {
#         'input_ids': torch.randint(0, tokenizer.vocab_size, (2, 10)),
#         'attention_mask': torch.ones(2, 10),
#         'labels': torch.randint(0, tokenizer.vocab_size, (2, 10))
#     }

#     device = next(model.parameters()).device
#     test_input = {k: v.to(device) for k, v in test_input.items()}

#     try:
#         scaled_logits, temperatures = model(test_input)

#         loss = scaled_logits.mean() + temperatures.mean()

#         loss.backward()

#         print("\nresults of the gradient checking:")
#         print("ATShead gradient status:")
#         for name, param in model.ats_head.named_parameters():
#             if param.grad is None:
#                 print(f"warn: {name} no gradient")
#             else:
#                 print(f"noraml: {name} Gradient norm: {param.grad.norm().item():.6f}")

#         print("\ncheck the gradient of all parameters:")
#         all_ok = True
#         for name, param in model.named_parameters():
#             if param.requires_grad and param.grad is None:
#                 print(f"warn: {name} need gradient")
#                 all_ok = False

#         if all_ok:
#             print("✓ we are best")

#         return True

#     except Exception as e:
#         print(f"\n❌ model gradinet check failed, error inf:\n{str(e)}")
#         print(f"\n error type: {type(e).__name__}")
#         return False
#     finally:
#         model.zero_grad()
# -

#

# def check_gradient_flow(model, tokenizer):

#     print("...")


#     test_input = {
#         'input_ids': torch.randint(0, tokenizer.vocab_size, (2, 10)),
#         'attention_mask': torch.ones(2, 10),
#         'labels': torch.randint(0, tokenizer.vocab_size, (2, 10))
#     }

#     device = next(model.parameters()).device
#     test_input = {k: v.to(device) for k, v in test_input.items()}

#     try:
#         scaled_logits, temperatures = model(test_input)

#         loss = scaled_logits.mean() + temperatures.mean()

#         loss.backward()

#         print("\nresults of the gradient checking:")
#         print("ATShead gradient status:")
#         for name, param in model.ats_head.named_parameters():
#             if param.grad is None:
#                 print(f"warn: {name} no gradient")
#             else:
#                 print(f"noraml: {name} Gradient norm: {param.grad.norm().item():.6f}")

#         print("\ncheck the gradient of all parameters:")
#         all_ok = True
#         for name, param in model.named_parameters():
#             if param.requires_grad and param.grad is None:
#                 print(f"warn: {name} need gradient")
#                 all_ok = False

#         if all_ok:
#             print("✓ we are best")

#         return True

#     except Exception as e:
#         print(f"\n❌ model gradinet check failed, error inf:\n{str(e)}")
#         print(f"\n error type: {type(e).__name__}")
#         return False
#     finally:
#         model.zero_grad()



# <!--     def forward(self, x):
#         outputs = self.model(
#         input_ids=x['input_ids'],
#         attention_mask=x.get('attention_mask', None),
#         labels=x.get('labels', None),
#         output_hidden_states=True,
#         return_dict=True
#     )
#         self.log_memory("After forward")
#         logits = outputs.logits
#         hidden_states = outputs.decoder_hidden_states[-1]
#         self.log_memory("After extracting decoder_hidden_states") -->

# <!--         temperatures = self.ats_head(hidden_states)
#         self.log_memory("After ats_head") -->

# <!--         scaled_logits = logits/temperatures
#         self.log_memory("After scaling logits") -->

# +
# #         return scaled_logits, temperatures
#     def forward(self, x):
# #         self.log_memory("Before forward")
#         outputs = self.model(
#             input_ids=x['input_ids'],
#             attention_mask=x.get('attention_mask', None),
#             labels=x.get('labels', None),
#             output_hidden_states=True,
#             return_dict=True
#         )
# #         self.log_memory("After forward")

#         logits = outputs.logits
# #         print(f"[Info] logits shape: {logits.shape}, dtype: {logits.dtype}")

#         hidden_states = outputs.decoder_hidden_states[-1].detach().clone().requires_grad_(True)
# #         self.log_memory("After extracting decoder_hidden_states")
# #         print(f"[Info] hidden_states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")

#         temperatures = self.ats_head(hidden_states)
# #         self.log_memory("After ats_head")
# #         print(f"[Info] temperatures shape: {temperatures.shape}, dtype: {temperatures.dtype}")

#         scaled_logits = logits / temperatures
# #         self.log_memory("After scaling logits")
# #         print(f"[Info] scaled_logits shape: {scaled_logits.shape}, dtype: {scaled_logits.dtype}")
#         torch.cuda.empty_cache()
#         return scaled_logits, temperatures



#     def configure_optimizers(self):
#         return transformer_optimizer_config(
#             self,
#             self.hparams.lr,
#             warmup_proportion=self.hparams.warmup_proportion,
#             weight_decay=self.hparams.wd,
#             decay_power=self.hparams.decay_power,
#         )

#     def training_step(self, x, idx):
# #         self.log_memory("Before training_step")
#         preds, temperatures = self.forward(x)
# #         self.log_memory("After forward in training_step")
        
#         stats = compute_loss_and_accuracy(preds, x["labels"], self.hparams.pad_token,alpha = 0.5)

#         self.log("ce", stats["loss"], prog_bar=True)
#         self.log("acc", stats["acc"], prog_bar=True)
#         self.log("exacts", stats["exacts"], prog_bar=True)
# #         self.log_memory("After loss computation")
        
#         if idx % 100 == 0:
#             torch.cuda.empty_cache()
#             gc.collect()
# #             self.log_memory("After empty_cache and gc.collect")
        
        
#         return stats["loss"]

#     def predict_step(self, x, idx):
#         preds, temperatures, logits_without_temperature_scaling = self.forward(x)
#         stats = compute_loss_and_accuracy(preds, x["labels"], self.hparams.pad_token,alpha=0.5)
#         return preds, x["input_ids"], x["labels"], stats["exacts"]

#     def validation_step(self, x, idx, dataloader_idx=0):
#         self.log_memory("Before validation_step")
#         preds, temperatures, logits_without_temperature_scaling = self.forward(x)
#         self.log_memory("After forward in validation_step")
#         stats = compute_loss_and_accuracy(preds, x["labels"], self.hparams.pad_token, alpha=0.5)

#         self.log("vce", stats["loss"], prog_bar=True)
#         self.log("vacc", stats["acc"], prog_bar=True)
#         self.log("vexacts", stats["exacts"], prog_bar=True)
#         self.log_memory("After loss computation in validation_step")
#         return stats["loss"]

# +
class T5Module(pl.LightningModule):
    def __init__(
        self,
        model,
        hidden_size,
        lr=0.0002,
        wd=1e-2,
        decay_power=-1,
        warmup_proportion=0.05,
        pad_token=None,
        train_mode = "t5",
        ats_head_layers=4  # 新增：ATS头的层数配置
        
    ):
        super().__init__()
        self.model = model
        self.ats_head = ATShead(hidden_size, num_layers=ats_head_layers)  # 传递层数
        self.save_hyperparameters(ignore=["model"])
        self.train_mode = train_mode
 
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.ats_head.parameters():
            param.requires_grad = True
            
        if train_mode == "t5":
            # Stage 1: 只训练主模型，冻结ATS头
            for param in self.ats_head.parameters():
                param.requires_grad = False
        else: 
            # Stage 2 (方案3): 同时训练主模型和ATS头，不冻结任何部分
            # 让主模型和ATS头都参与训练，通过学习率差异控制训练强度
            pass  # 保持所有参数都可训练

    
    def log_memory(self, step_name):
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
#         print(f"[Memory] {step_name}: Allocated: {allocated / (1024 ** 2):.2f} MB, Reserved: {reserved / (1024 ** 2):.2f} MB")
    def forward(self, x):
#         self.log_memory("Before forward")
        outputs = self.model(
            input_ids=x['input_ids'],
            attention_mask=x.get('attention_mask', None),
            labels=x.get('labels', None),
            output_hidden_states=True,
            return_dict=True
        )
#         self.log_memory("After forward")

        logits = outputs.logits
#         print(f"[Info] logits shape: {logits.shape}, dtype: {logits.dtype}")
        if self.train_mode =="t5":
            hidden_states = outputs.decoder_hidden_states[-1]
            logits = outputs.logits
            temperatures = 1
            scaled_logits = logits
        else:
            hidden_states = outputs.decoder_hidden_states[-1].detach().clone().requires_grad_(True)
    #         self.log_memory("After extracting decoder_hidden_states")
    #         print(f"[Info] hidden_states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
            logits = outputs.logits
            temperatures = self.ats_head(hidden_states)
    #         self.log_memory("After ats_head")
    #         print(f"[Info] temperatures shape: {temperatures.shape}, dtype: {temperatures.dtype}")

            scaled_logits = logits / temperatures
    #         self.log_memory("After scaling logits")
    #         print(f"[Info] scaled_logits shape: {scaled_logits.shape}, dtype: {scaled_logits.dtype}")
        torch.cuda.empty_cache()
        return scaled_logits, temperatures, logits



    def configure_optimizers(self):
        return transformer_optimizer_config(
            self,
            self.hparams.lr,
            warmup_proportion=self.hparams.warmup_proportion,
            weight_decay=self.hparams.wd,
            decay_power=self.hparams.decay_power,
            train_mode = self.train_mode
        )

    def training_step(self, x, idx ):
#         self.log_memory("Before training_step")
        self.train()
        self.model.train()
        preds, temperatures, logits_without_temperature_scaling= self.forward(x)
        train_mode = self.train_mode
#         self.log_memory("After forward in training_step")
        
        if train_mode == "t5":
            stats = compute_loss_and_accuracy_first(preds, x["labels"], self.hparams.pad_token)

            self.log("ce", stats["loss"], prog_bar=True)
            self.log("acc", stats["acc"], prog_bar=True)
            self.log("exacts", stats["exacts"], prog_bar=True)
        else:
            # 使用强化的ATS专用损失函数
            stats = compute_loss_and_accuracy_ats(
                preds,  # 温度缩放后的logits
                logits_without_temperature_scaling,  # 原始logits
                temperatures,  # 预测的温度值
                x["labels"],  # 目标labels
                self.hparams.pad_token,  # padding token id
                alpha=0.5,
                temp_weight=1.5  # 强化的温度正则化权重
            )

            # 记录详细的损失组件
            self.log("ce", stats["ce_loss"], prog_bar=True)
            self.log("temp_reg", stats["temp_reg_loss"], prog_bar=True)
            self.log("total_loss", stats["loss"], prog_bar=True)
            self.log("acc", stats["acc"], prog_bar=True)
            self.log("exacts", stats["exacts"], prog_bar=True)

        
        
        return stats["loss"]

    def predict_step(self, x, idx):
        preds, temperatures, logits_without_temperature_scaling = self.forward(x)
        
        if self.train_mode == "t5":
            # T5模式使用标准损失函数
            stats = compute_loss_and_accuracy(
                preds,  # 预测logits
                logits_without_temperature_scaling,  # 原始logits
                x["labels"],  # 目标labels
                self.hparams.pad_token,  # padding token id
                alpha=0.5
            )
        else:
            # ATS模式使用温度正则化损失函数
            stats = compute_loss_and_accuracy_ats(
                preds,  # 温度缩放后的logits
                logits_without_temperature_scaling,  # 原始logits
                temperatures,  # 预测的温度值
                x["labels"],  # 目标labels
                self.hparams.pad_token,  # padding token id
                alpha=0.5,
                temp_weight=1.5
            )
        return preds, x["input_ids"], x["labels"], stats["exacts"]

    def validation_step(self, x, idx, dataloader_idx=0):
        self.log_memory("Before validation_step")
        preds, temperatures, logits_without_temperature_scaling = self.forward(x)
        self.log_memory("After forward in validation_step")
        
        if self.train_mode == "t5":
            # T5模式使用标准损失函数
            stats = compute_loss_and_accuracy(
                preds,  # 预测logits
                logits_without_temperature_scaling,  # 原始logits
                x["labels"],  # 目标labels
                self.hparams.pad_token,  # padding token id
                alpha=0.5
            )
        else:
            # ATS模式使用温度正则化损失函数
            stats = compute_loss_and_accuracy_ats(
                preds,  # 温度缩放后的logits
                logits_without_temperature_scaling,  # 原始logits
                temperatures,  # 预测的温度值
                x["labels"],  # 目标labels
                self.hparams.pad_token,  # padding token id
                alpha=0.5,
                temp_weight=1.5
            )

        self.log("vce", stats["loss"], prog_bar=True)
        self.log("vacc", stats["acc"], prog_bar=True)
        self.log("vexacts", stats["exacts"], prog_bar=True)
        self.log_memory("After loss computation in validation_step")
        return stats["loss"]
    

 
def finetune_tf_lm(
    model_name_or_path,
    tokenizer,
    train_pairs,
    train_full,
    val_pairs,
    load_from: str = None,
    do_fit: bool = True,
    root_dir: str = "logs"
):
    pl.seed_everything(0)
    config =AutoConfig.from_pretrained(
        model_name_or_path,
        output_hidden_states=True,
        return_dict = True,
        use_cache=False
    )
    model = T5ForConditionalGeneration.from_pretrained(
        model_name_or_path,
        config =config,
        cache_dir="/tmp/hf"
    )
    hidden_size = model.config.d_model
    harness = T5Module(model, 
                       pad_token=tokenizer.pad_token_id,
                       hidden_size=hidden_size,
                       train_mode="t5")

    if load_from:
        harness.load_state_dict(torch.load(load_from)["state_dict"])
    

#     if not check_gradient_flow(harness, tokenizer):
#         print("model gradinet check failed")
#         return None, None, None
    
    torch.set_float32_matmul_precision("medium")
    
    # 检测设备并设置相应的配置
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
        print("使用GPU训练")
    else:
        accelerator = "cpu"
        devices = 1
        precision = "32"
        print("使用CPU训练")
    
    trainer = pl.Trainer(
        max_steps=3000,
        val_check_interval=1.0,
        num_sanity_val_steps=1,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        accumulate_grad_batches=1,
        enable_checkpointing=False,
        default_root_dir=f"{root_dir}/stage1"
        
    )
    if do_fit:
        trainer.fit(
            harness,
            DataLoader(
                TokenizerPairIterableDataset(list(train_pairs), tokenizer),
                batch_size=52,
            ),
            DataLoader(
                TokenizerPairIterableDataset(list(val_pairs), tokenizer), batch_size=52
            ),
        )
#         checkpoint_path = "logs/stage1/t5_trained.pt"
#         checkpoint = torch.load(checkpoint_path, map_location="cpu")

    
#         pl_checkpoint = {
#             "state_dict": checkpoint["state_dict"],
#             "pytorch-lightning_version": "2.2.2", 
#         }


#         torch.save(pl_checkpoint, "logs/stage1/t5_trained_fixed.ckpt")
        torch.save({
            "state_dict": harness.state_dict(),
            "pytorch-lightning_version": "2.2.2"
        }, f"{root_dir}/stage1/final_model_new_temperature_newloss_layer4_stage1.ckpt")
        
        harness_stage2 = T5Module(model,
                                pad_token=tokenizer.pad_token_id,
                                hidden_size=hidden_size,
                                train_mode="ats") 
        

#         harness_stage2.load_state_dict(torch.load(checkpoint_path)["state_dict"])
        harness_stage2.load_state_dict(torch.load(f"{root_dir}/stage1/final_model_new_temperature_newloss_layer4_stage1.ckpt")["state_dict"])
        trainer_stage2 = pl.Trainer(
            max_steps=15000,  # Stage 2: 显著增加到10000步
            val_check_interval=500, # 修改为整数：每500个训练批次验证一次
            num_sanity_val_steps=1,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            accumulate_grad_batches=1,
            default_root_dir=f"{root_dir}/stage2",
            enable_checkpointing=False, # 保持不保存中间checkpoint以加速
        )
        
        
        trainer_stage2.fit(
            harness_stage2,
            DataLoader(
                TokenizerPairIterableDataset(list(train_full), tokenizer),
                batch_size=52,
            ),
            DataLoader(
                TokenizerPairIterableDataset(list(val_pairs), tokenizer), 
                batch_size=52
            )
            
        )
        torch.save({
            "state_dict": harness_stage2.state_dict(),
            "pytorch-lightning_version": "2.2.2"
        }, f"{root_dir}/stage2/final_model_new_temperature_newloss_layer4_stage2.ckpt")
        
        return trainer_stage2, model, harness_stage2
    return trainer, model, harness
# -


