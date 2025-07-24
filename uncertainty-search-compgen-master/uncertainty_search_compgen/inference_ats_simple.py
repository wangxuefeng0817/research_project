import sys
import time
import torch
import numpy as np
import torch.nn.functional as F
from IPython.display import clear_output


@torch.inference_mode()
def ats_uncertainty_guided_search(
    harness, batch, tokenizer, k=128, threshold=0.8, keep_n=8, pick=2, out_length=448
):
    """
    使用ATS头的uncertainty来指导beam search，但保持原始uncertainty guided search的逻辑
    """
    inp_batch = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()
    batch_size = inp_batch.shape[0]
    tgt = (
        torch.ones_like(inp_batch[:, :1]) * harness.model.config.decoder_start_token_id
    )
    overall_entr = torch.zeros_like(tgt).to(torch.float).flatten()
    done_beams = torch.zeros_like(inp_batch[:, :1]).to(torch.bool)

    # EOS token处理
    done_seq = torch.tensor(
        tokenizer.encode("[eos]")[1:-1], device=tgt.device, dtype=tgt.dtype
    )
    done_seq_alt = torch.tensor(
        tokenizer.encode(" [eos]")[1:-1], device=tgt.device, dtype=tgt.dtype
    )
    batch_indices = torch.arange(done_beams.shape[0], device="cuda", dtype=torch.int)

    harness.cuda()
    harness.eval()

    for i in range(k):
        if tgt.shape[-1] >= done_seq.shape[0]:
            done_beams = (
                done_beams
                | (tgt[:, -done_seq.shape[0] :] == done_seq[None]).all(dim=-1)[:, None]
                | (tgt[:, -done_seq_alt.shape[0] :] == done_seq_alt[None]).all(dim=-1)[
                    :, None
                ]
            )

        # 🔥 关键改进：使用ATS头预测uncertainty
        if harness.train_mode == "ats":
            temp_batch = {
                "input_ids": inp_batch,
                "labels": tgt
            }
            scaled_logits, ats_temperatures, original_logits = harness.forward(temp_batch)
            
            # 使用原始logits计算概率分布
            predicted_out_logits = original_logits[:, -1:]
            
            # 使用ATS头的temperature作为uncertainty的proxy
            if hasattr(ats_temperatures, 'mean'):
                ats_uncertainty = ats_temperatures[:, -1, :].mean(dim=-1)  # (batch_size,)
            else:
                ats_uncertainty = torch.ones(batch_size, device=inp_batch.device) * ats_temperatures
            
            # 将ATS uncertainty映射到entropy的量级（用于与阈值比较）
            # 基于诊断结果，ATS范围是0.7-1.0，entropy范围是0.006-4.6
            # 我们将ATS uncertainty放大到合适的范围
            entr = (ats_uncertainty - 0.5) * 10.0  # 映射到0-5的范围
            entr = torch.clamp(entr, 0, 10)  # 限制范围
            
            # print(f"Step {i+1}: ATS uncertainty={ats_uncertainty.mean().item():.4f}, mapped entropy={entr.mean().item():.4f}")
            
        else:
            # 使用标准模型
            predicted_out_logits = harness.model(
                **{
                    "input_ids": inp_batch,
                    "decoder_input_ids": tgt,
                    "attention_mask": torch.ones_like(inp_batch),
                    "decoder_attention_mask": torch.ones_like(tgt),
                }
            ).logits[:, -1:]
            
            # 计算标准entropy
            out_logits_p = predicted_out_logits.softmax(dim=-1)
            entr = torch.special.entr(out_logits_p).sum(dim=-1)
            
            # print(f"Step {i+1}: Standard entropy={entr.mean().item():.4f}")

        out_logits = predicted_out_logits

        # 为已完成的beam设置EOS token
        out_logits = (
            out_logits * ~done_beams[..., None]
            + F.one_hot(torch.tensor(tokenizer.eos_token_id), out_logits.shape[-1])[
                None, None, :
            ].to(out_logits.device)
            * done_beams[..., None]
            * 9999999
        )

        # 🔥 使用mapped entropy进行beam expansion决策
        out_logits_p = out_logits.softmax(dim=-1)
        
        # 使用之前计算的ATS映射entropy，确保维度匹配
        if entr.dim() == 1:
            entr = entr[..., None]  # 添加维度以匹配done_beams
        entr[entr.isnan() | done_beams] = 0.0

        # 原始逻辑：基于entropy决定beam expansion
        topks = out_logits_p.topk(pick).indices
        tops = out_logits_p.argmax(dim=-1)[..., None]
        
        # 修复广播问题：确保entr的维度与topks/tops匹配
        select_mask_topks = (entr > threshold).unsqueeze(-1).expand_as(topks)
        select_mask_tops = (~(entr > threshold)).unsqueeze(-1).expand_as(tops)

        # 填充已完成的beam
        cat_tops = torch.cat([topks, tops], dim=-1)
        cat_tops = ~done_beams[:, None] * cat_tops + (
            done_beams[:, None] * torch.ones_like(cat_tops) * tokenizer.eos_token_id
        )

        # 构建masks - 恢复原始逻辑
        select_mask_tops = torch.cat([select_mask_topks, select_mask_tops], dim=-1)
        # print(f"Debug: after cat - select_mask_tops.shape={select_mask_tops.shape}")
        
        beam_ids = torch.arange(
            inp_batch.shape[0], device=inp_batch.device, dtype=inp_batch.dtype
        )[:, None, None].expand(-1, 1, select_mask_tops.shape[-1])
        # print(f"Debug: beam_ids.shape={beam_ids.shape}")
        
        select_beams = beam_ids[select_mask_tops].flatten()
        # print(f"Debug: select_beams.shape={select_beams.shape}")
        
        select_nexts = cat_tops[select_mask_tops].flatten()
        # print(f"Debug: select_nexts.shape={select_nexts.shape}")
        
        # 先尝试最简单的方法
        # print(f"Debug: before entr_nexts - entr.shape={entr.shape}, select_mask_tops.shape={select_mask_tops.shape}")
        # print(f"Debug: select_beams values: {select_beams[:10]}")  # 看看前10个值
        
        # 使用select_beams来索引entropy值
        entr_nexts = entr.squeeze(-1)[select_beams]
        # print(f"Debug: entr_nexts.shape={entr_nexts.shape}")

        # 更新状态
        inp_batch = inp_batch[select_beams]
        labels = labels[select_beams]
        tgt = torch.cat([tgt[select_beams], select_nexts[:, None]], dim=-1)
        overall_entr = overall_entr[select_beams] + entr_nexts
        done_beams = done_beams[select_beams]
        batch_indices = batch_indices[select_beams]

        # Beam pruning
        excess_entropy_beams = (overall_entr / tgt.shape[1]) > 100
        excess_entropy_beams ^= excess_entropy_beams.all()[None]

        tgt = tgt[~excess_entropy_beams]
        done_beams = done_beams[~excess_entropy_beams]
        inp_batch = inp_batch[~excess_entropy_beams]
        labels = labels[~excess_entropy_beams]
        batch_indices = batch_indices[~excess_entropy_beams]
        overall_entr = overall_entr[~excess_entropy_beams]

        # 保留top keep_n个beams
        def keep_n_beams(arr):
            batches_top_n = []
            for i in range(batch_size):
                top_n_mask = (overall_entr[batch_indices == i].flatten()).argsort()[
                    :keep_n
                ]
                batches_top_n.append(arr[batch_indices == i][top_n_mask])
            return torch.cat(batches_top_n, dim=0)

        tgt = keep_n_beams(tgt)
        done_beams = keep_n_beams(done_beams)
        inp_batch = keep_n_beams(inp_batch)
        labels = keep_n_beams(labels)
        next_batch_indices = keep_n_beams(batch_indices)
        overall_entr = keep_n_beams(overall_entr)
        batch_indices = next_batch_indices

        if done_beams.all(dim=0)[0].item():
            break

        # 打印统计信息
        expand_count = (entr.squeeze() > threshold).sum().item()
        # print(f"Step {i+1}: threshold={threshold}, expand={expand_count}/{entr.shape[0]}")

    # 填充输出
    padding = (
        0,
        out_length - tgt.size(-1),
    )
    tgt = F.pad(tgt, padding, mode="constant", value=tokenizer.eos_token_id)

    return [
        list(
            zip(
                inp_batch.cpu().numpy()[batch_indices.cpu().numpy() == i],
                tgt.cpu().numpy()[batch_indices.cpu().numpy() == i],
                labels.cpu().numpy()[batch_indices.cpu().numpy() == i],
            )
        )
        for i in range(batch_size)
    ]


def ats_uncertainty_guided_search_wrapper(
    harness, batch, tokenizer, k=32, keep_n=3, out_length=64, threshold=0.8, **kwargs
):
    """
    包装器，转换输出格式为标准格式
    """
    out = ats_uncertainty_guided_search(
        harness,
        batch,
        tokenizer,
        k=k,
        keep_n=keep_n,
        out_length=out_length + 1,
        threshold=threshold,
        **kwargs,
    )
    out_logits = []
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    for b in out:
        _batch = []
        for beam in b[:keep_n]:
            tgt_seq = beam[1][1:]  # tgt，移除start token
            if len(tgt_seq) > out_length:
                tgt_seq = tgt_seq[:out_length]
            _batch.append(tgt_seq)
        
        # 确保有足够的beams
        while len(_batch) < keep_n:
            _batch.append(np.full(out_length, pad_token_id, dtype=np.int64))

        # 确保所有序列长度一致
        for i in range(len(_batch)):
            if len(_batch[i]) < out_length:
                padding_length = out_length - len(_batch[i])
                _batch[i] = np.concatenate([_batch[i], np.full(padding_length, pad_token_id, dtype=np.int64)])
            elif len(_batch[i]) > out_length:
                _batch[i] = _batch[i][:out_length]

        out_logits.append(np.vstack(_batch))

    out_logits = torch.tensor(np.array(out_logits), dtype=torch.int64).permute(
        (0, 2, 1)
    )

    return out_logits