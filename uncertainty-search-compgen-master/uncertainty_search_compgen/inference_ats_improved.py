import sys
import time
import torch
import numpy as np
import torch.nn.functional as F
from IPython.display import clear_output


@torch.inference_mode()
def ats_guided_beam_search(
    harness, batch, tokenizer, k=128, keep_n=8, pick=2, out_length=448, 
    ats_threshold_percentile=75, use_adaptive_threshold=True
):
    """
    ATS-Guided Beam Search: 直接使用ATS头的不确定性预测来指导beam expansion
    
    关键改进：
    1. 使用ATS头的预测作为uncertainty metric，而不是简单的temperature scaling
    2. 动态阈值：基于ATS头预测的分布来设置阈值
    3. 将ATS头的输出映射为uncertainty score
    
    Args:
        ats_threshold_percentile: 使用ATS预测的第X百分位数作为阈值
        use_adaptive_threshold: 是否使用自适应阈值
    """
    inp_batch = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()
    batch_size = inp_batch.shape[0]
    tgt = (
        torch.ones_like(inp_batch[:, :1]) * harness.model.config.decoder_start_token_id
    )
    overall_uncertainty = torch.zeros_like(tgt).to(torch.float).flatten()
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

    # 用于收集统计信息的列表
    ats_predictions_history = []

    for i in range(k):
        if tgt.shape[-1] >= done_seq.shape[0]:
            # 标记完成的序列
            done_beams = (
                done_beams
                | (tgt[:, -done_seq.shape[0] :] == done_seq[None]).all(dim=-1)[:, None]
                | (tgt[:, -done_seq_alt.shape[0] :] == done_seq_alt[None]).all(dim=-1)[
                    :, None
                ]
            )

        # 🔥 关键改进：使用ATS头来预测不确定性
        if harness.train_mode == "ats":
            # 使用ATS头预测
            temp_batch = {
                "input_ids": inp_batch,
                "labels": tgt
            }
            scaled_logits, ats_temperatures, original_logits = harness.forward(temp_batch)
            
            # 将ATS头的temperature转换为uncertainty score
            # 策略1: 使用temperature本身作为uncertainty（温度越高越不确定）
            if hasattr(ats_temperatures, 'mean'):
                ats_uncertainty = ats_temperatures[:, -1, :].mean(dim=-1)  # 取最后一个位置的平均温度
            else:
                ats_uncertainty = torch.ones(batch_size, device=inp_batch.device) * ats_temperatures
            
            # 策略2: 也可以使用temperature的方差作为uncertainty
            # if hasattr(ats_temperatures, 'var'):
            #     ats_uncertainty = ats_temperatures[:, -1, :].var(dim=-1)
            
            # 使用原始logits（不经过温度缩放）
            out_logits = original_logits[:, -1:]
            
            # 收集ATS预测用于统计
            ats_predictions_history.append(ats_uncertainty.cpu().numpy())
            
            print(f"Step {i+1}: ATS uncertainty mean={ats_uncertainty.mean().item():.4f}")
            
        else:
            # 如果不是ATS模式，回退到标准模型
            predicted_out_logits = harness.model(
                **{
                    "input_ids": inp_batch,
                    "decoder_input_ids": tgt,
                    "attention_mask": torch.ones_like(inp_batch),
                    "decoder_attention_mask": torch.ones_like(tgt),
                }
            ).logits[:, -1:]
            
            out_logits = predicted_out_logits
            # 使用entropy作为fallback
            out_logits_p = out_logits.softmax(dim=-1)
            ats_uncertainty = torch.special.entr(out_logits_p).sum(dim=-1)
            ats_predictions_history.append(ats_uncertainty.cpu().numpy())

        # 🔥 动态阈值计算
        if use_adaptive_threshold and len(ats_predictions_history) >= 3:
            # 基于历史预测计算阈值
            all_predictions = np.concatenate(ats_predictions_history)
            threshold = np.percentile(all_predictions, ats_threshold_percentile)
        else:
            # 固定阈值策略：基于ATS头的输出范围
            if harness.train_mode == "ats":
                # 对于ATS头，使用0.8作为阈值（基于诊断结果）
                threshold = 0.8
            else:
                # 对于entropy，使用0.4作为阈值
                threshold = 0.4

        # 为已完成的beam设置EOS token
        out_logits = (
            out_logits * ~done_beams[..., None]
            + F.one_hot(torch.tensor(tokenizer.eos_token_id), out_logits.shape[-1])[
                None, None, :
            ].to(out_logits.device)
            * done_beams[..., None]
            * 9999999
        )

        # 🔥 核心决策：基于ATS uncertainty决定beam expansion
        out_logits_p = out_logits.softmax(dim=-1)
        
        # 如果uncertainty > threshold，选择top "pick" beams并expansion
        # 否则只保留top1
        topks = out_logits_p.topk(pick).indices  # (batch_size, pick)
        tops = out_logits_p.argmax(dim=-1)[..., None]  # (batch_size, 1)
        
        # 基于uncertainty决定每个样本选择哪些tokens
        select_mask_topks = (ats_uncertainty > threshold)[:, None].expand(-1, pick).bool()
        select_mask_tops = (~(ats_uncertainty > threshold))[:, None].expand(-1, 1).bool()

        # 填充已完成的beam
        cat_tops = torch.cat([topks, tops], dim=-1)  # (batch_size, pick+1)
        cat_tops = ~done_beams.expand(-1, cat_tops.shape[-1]) * cat_tops + (
            done_beams.expand(-1, cat_tops.shape[-1]) * torch.ones_like(cat_tops) * tokenizer.eos_token_id
        )

        # 构建selection masks
        select_mask_all = torch.cat([select_mask_topks, select_mask_tops], dim=-1)  # (batch_size, pick+1)
        
        # 为每个batch样本创建beam indices
        beam_ids = torch.arange(
            inp_batch.shape[0], device=inp_batch.device, dtype=inp_batch.dtype
        )[:, None].expand(-1, select_mask_all.shape[-1])  # (batch_size, pick+1)
        
        select_beams = beam_ids[select_mask_all].flatten()
        select_nexts = cat_tops[select_mask_all].flatten()
        
        # 更新uncertainty累积
        uncertainty_nexts = ats_uncertainty[:, None].expand(-1, select_mask_all.shape[-1])[
            select_mask_all
        ].flatten()

        # 更新状态
        inp_batch = inp_batch[select_beams]
        labels = labels[select_beams]
        tgt = torch.cat([tgt[select_beams], select_nexts[:, None]], dim=-1)
        overall_uncertainty = overall_uncertainty[select_beams] + uncertainty_nexts
        done_beams = done_beams[select_beams]
        batch_indices = batch_indices[select_beams]

        # Beam pruning：基于累积uncertainty
        if harness.train_mode == "ats":
            # 对于ATS头，保留uncertainty适中的beams
            avg_uncertainty = overall_uncertainty / tgt.shape[1]
            excess_uncertainty_beams = avg_uncertainty > 2.0  # 基于ATS头的range调整
        else:
            # 对于entropy，使用原来的逻辑
            excess_uncertainty_beams = (overall_uncertainty / tgt.shape[1]) > 100

        # 不要删除所有beams
        excess_uncertainty_beams ^= excess_uncertainty_beams.all()[None]

        tgt = tgt[~excess_uncertainty_beams]
        done_beams = done_beams[~excess_uncertainty_beams]
        inp_batch = inp_batch[~excess_uncertainty_beams]
        labels = labels[~excess_uncertainty_beams]
        batch_indices = batch_indices[~excess_uncertainty_beams]
        overall_uncertainty = overall_uncertainty[~excess_uncertainty_beams]

        # 保留top keep_n个beams
        def keep_n_beams(arr):
            batches_top_n = []
            for i in range(batch_size):
                if (batch_indices == i).sum() > 0:
                    top_n_mask = (overall_uncertainty[batch_indices == i].flatten()).argsort()[
                        :keep_n
                    ]
                    batches_top_n.append(arr[batch_indices == i][top_n_mask])
            return torch.cat(batches_top_n, dim=0) if batches_top_n else arr[:0]

        tgt = keep_n_beams(tgt)
        done_beams = keep_n_beams(done_beams)
        inp_batch = keep_n_beams(inp_batch)
        labels = keep_n_beams(labels)
        next_batch_indices = keep_n_beams(batch_indices)
        overall_uncertainty = keep_n_beams(overall_uncertainty)
        batch_indices = next_batch_indices

        if len(batch_indices) == 0 or done_beams.all(dim=0)[0].item():
            break

        # 打印统计信息
        expand_count = (ats_uncertainty > threshold).sum().item()
        print(f"Step {i+1}: threshold={threshold:.4f}, expand={expand_count}/{ats_uncertainty.shape[0]}")

    # 最终输出处理
    padding = (0, out_length - tgt.size(-1))
    tgt = F.pad(tgt, padding, mode="constant", value=tokenizer.eos_token_id)

    # 返回格式与原始方法一致
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


def ats_guided_beam_search_wrapper(
    harness, batch, tokenizer, k=32, keep_n=3, out_length=64, **kwargs
):
    """
    包装器，转换输出格式为标准格式
    Return shape: (BATCH, SEQ_LEN, @K)
    """
    out = ats_guided_beam_search(
        harness,
        batch,
        tokenizer,
        k=k,
        keep_n=keep_n,
        out_length=out_length,
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


@torch.inference_mode()
def hybrid_uncertainty_beam_search(
    harness, batch, tokenizer, k=128, keep_n=8, pick=2, out_length=448,
    ats_weight=0.7, entropy_weight=0.3, threshold=0.4
):
    """
    混合不确定性beam search：结合ATS头和entropy
    
    Args:
        ats_weight: ATS头预测的权重
        entropy_weight: entropy的权重
        threshold: 混合uncertainty的阈值
    """
    inp_batch = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()
    batch_size = inp_batch.shape[0]
    tgt = (
        torch.ones_like(inp_batch[:, :1]) * harness.model.config.decoder_start_token_id
    )
    overall_uncertainty = torch.zeros_like(tgt).to(torch.float).flatten()
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

        # 🔥 混合uncertainty计算
        if harness.train_mode == "ats":
            # 获取ATS头预测
            temp_batch = {
                "input_ids": inp_batch,
                "labels": tgt
            }
            scaled_logits, ats_temperatures, original_logits = harness.forward(temp_batch)
            
            # ATS uncertainty
            if hasattr(ats_temperatures, 'mean'):
                ats_uncertainty = ats_temperatures[:, -1, :].mean(dim=-1)
            else:
                ats_uncertainty = torch.ones(batch_size, device=inp_batch.device) * ats_temperatures
            
            # 归一化ATS uncertainty到[0,1]
            ats_uncertainty_norm = (ats_uncertainty - 0.5) / 1.5  # 假设ATS范围是[0.5, 2.0]
            ats_uncertainty_norm = torch.clamp(ats_uncertainty_norm, 0, 1)
            
            out_logits = original_logits[:, -1:]
        else:
            # 标准模型
            predicted_out_logits = harness.model(
                **{
                    "input_ids": inp_batch,
                    "decoder_input_ids": tgt,
                    "attention_mask": torch.ones_like(inp_batch),
                    "decoder_attention_mask": torch.ones_like(tgt),
                }
            ).logits[:, -1:]
            
            out_logits = predicted_out_logits
            ats_uncertainty_norm = torch.zeros(batch_size, device=inp_batch.device)

        # 计算entropy
        out_logits_p = out_logits.softmax(dim=-1)
        entropy = torch.special.entr(out_logits_p).sum(dim=-1)
        
        # 归一化entropy到[0,1]
        entropy_norm = torch.clamp(entropy / 10.0, 0, 1)  # 假设entropy最大值是10
        
        # 🔥 混合uncertainty score
        combined_uncertainty = ats_weight * ats_uncertainty_norm + entropy_weight * entropy_norm
        
        print(f"Step {i+1}: ATS={ats_uncertainty_norm.mean().item():.4f}, "
              f"Entropy={entropy_norm.mean().item():.4f}, "
              f"Combined={combined_uncertainty.mean().item():.4f}")

        # 设置EOS token
        out_logits = (
            out_logits * ~done_beams[..., None]
            + F.one_hot(torch.tensor(tokenizer.eos_token_id), out_logits.shape[-1])[
                None, None, :
            ].to(out_logits.device)
            * done_beams[..., None]
            * 9999999
        )

        # 基于混合uncertainty决定beam expansion
        topks = out_logits_p.topk(pick).indices  # (batch_size, pick)
        tops = out_logits_p.argmax(dim=-1)[..., None]  # (batch_size, 1)
        
        # 基于uncertainty决定每个样本选择哪些tokens
        select_mask_topks = (combined_uncertainty > threshold)[:, None].expand(-1, pick).bool()
        select_mask_tops = (~(combined_uncertainty > threshold))[:, None].expand(-1, 1).bool()

        # 后续处理与ats_guided_beam_search相同
        cat_tops = torch.cat([topks, tops], dim=-1)  # (batch_size, pick+1)
        cat_tops = ~done_beams.expand(-1, cat_tops.shape[-1]) * cat_tops + (
            done_beams.expand(-1, cat_tops.shape[-1]) * torch.ones_like(cat_tops) * tokenizer.eos_token_id
        )

        select_mask_all = torch.cat([select_mask_topks, select_mask_tops], dim=-1)  # (batch_size, pick+1)
        
        # 为每个batch样本创建beam indices
        beam_ids = torch.arange(
            inp_batch.shape[0], device=inp_batch.device, dtype=inp_batch.dtype
        )[:, None].expand(-1, select_mask_all.shape[-1])  # (batch_size, pick+1)
        
        select_beams = beam_ids[select_mask_all].flatten()
        select_nexts = cat_tops[select_mask_all].flatten()
        
        uncertainty_nexts = combined_uncertainty[:, None].expand(-1, select_mask_all.shape[-1])[
            select_mask_all
        ].flatten()

        # 更新状态
        inp_batch = inp_batch[select_beams]
        labels = labels[select_beams]
        tgt = torch.cat([tgt[select_beams], select_nexts[:, None]], dim=-1)
        overall_uncertainty = overall_uncertainty[select_beams] + uncertainty_nexts
        done_beams = done_beams[select_beams]
        batch_indices = batch_indices[select_beams]

        # Beam pruning
        excess_uncertainty_beams = (overall_uncertainty / tgt.shape[1]) > 1.0
        excess_uncertainty_beams ^= excess_uncertainty_beams.all()[None]

        tgt = tgt[~excess_uncertainty_beams]
        done_beams = done_beams[~excess_uncertainty_beams]
        inp_batch = inp_batch[~excess_uncertainty_beams]
        labels = labels[~excess_uncertainty_beams]
        batch_indices = batch_indices[~excess_uncertainty_beams]
        overall_uncertainty = overall_uncertainty[~excess_uncertainty_beams]

        # 保留top keep_n个beams
        def keep_n_beams(arr):
            batches_top_n = []
            for i in range(batch_size):
                if (batch_indices == i).sum() > 0:
                    top_n_mask = (overall_uncertainty[batch_indices == i].flatten()).argsort()[
                        :keep_n
                    ]
                    batches_top_n.append(arr[batch_indices == i][top_n_mask])
            return torch.cat(batches_top_n, dim=0) if batches_top_n else arr[:0]

        tgt = keep_n_beams(tgt)
        done_beams = keep_n_beams(done_beams)
        inp_batch = keep_n_beams(inp_batch)
        labels = keep_n_beams(labels)
        next_batch_indices = keep_n_beams(batch_indices)
        overall_uncertainty = keep_n_beams(overall_uncertainty)
        batch_indices = next_batch_indices

        if len(batch_indices) == 0 or done_beams.all(dim=0)[0].item():
            break

        # 打印统计信息
        expand_count = (combined_uncertainty > threshold).sum().item()
        print(f"Step {i+1}: threshold={threshold:.4f}, expand={expand_count}/{combined_uncertainty.shape[0]}")

    # 最终输出处理
    padding = (0, out_length - tgt.size(-1))
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


def hybrid_uncertainty_beam_search_wrapper(
    harness, batch, tokenizer, k=32, keep_n=3, out_length=64, **kwargs
):
    """
    混合uncertainty beam search包装器
    """
    out = hybrid_uncertainty_beam_search(
        harness,
        batch,
        tokenizer,
        k=k,
        keep_n=keep_n,
        out_length=out_length,
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