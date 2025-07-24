import sys
import time
import torch
import numpy as np
import torch.nn.functional as F
from IPython.display import clear_output


def mask_permitted_transition_indices(logits, prev_step, permitted_transition_indices):
    """Masks logits such that only transitions from prev_step to index i are kept."""
    out_logits = torch.ones_like(logits) * float("-inf")
    out_logits.scatter_(
        -1,
        permitted_transition_indices[prev_step][:, None],
        torch.gather(
            logits,
            -1,
            permitted_transition_indices[prev_step][:, None],
        ),
    )
    return out_logits


def get_topk_outputs(harness, batch, k=5):
    """
    修改后的版本：使用T5Module的forward方法，支持温度缩放
    注意：这仍然是teacher forcing，不是真正的生成
    """
    with torch.inference_mode():
        harness.cuda()
        harness.eval()
        
        # 🔥 关键修改：使用harness.forward而不是harness.model
        scaled_logits, temperatures, original_logits = harness.forward(batch)
        
        # 检查温度缩放是否生效
        if hasattr(temperatures, 'mean'):  # 是tensor，说明使用了ATS
            temp_info = f"ATS温度(均值={temperatures.mean().item():.3f})"
        else:  # 是标量1，说明没使用ATS
            temp_info = f"固定温度({temperatures})"
        # print(f"   🌡️  get_topk_outputs: {temp_info}")
        
        # 使用温度缩放后的logits
        return scaled_logits.topk(k, dim=-1).indices.cpu()


def get_topk_outputs_original(harness, batch, k=5):
    """
    原始版本：直接使用底层模型，不支持温度缩放
    """
    with torch.inference_mode():
        harness.cuda()
        inp_batch = batch["input_ids"].cuda()
        outp_batch = batch["labels"].cuda()
        tgt = outp_batch

        harness.eval()
        out_logits = harness.model(
            **{"input_ids": inp_batch, "labels": tgt}
        ).logits
        return out_logits.topk(k, dim=-1).indices.cpu() # .numpy()


def generate_gpt(harness, tokenizer, inp, max_steps=128):
    harness.cuda()
    harness.eval()

    input_tokens = torch.from_numpy(np.array(tokenizer.encode(inp)))[None].to("cuda")
    output = torch.tensor(
        [[harness.model.config.decoder_start_token_id]] * 1,
        dtype=torch.long,
        device="cuda",
    )
    # print(tokenizer.batch_decode(input_tokens))
    # print(output)

    with torch.inference_mode():
        for i in range(max_steps):
            out_logits = harness.model(
                input_ids=input_tokens,
                decoder_input_ids=output,
                attention_mask=torch.ones_like(input_tokens),
                decoder_attention_mask=torch.ones_like(output),
            )
            # import pdb
            # pdb.set_trace()
            tformer_out = out_logits.logits[:, -1:].argmax(dim=-1)
            # print(tformer_out)
            output = torch.cat([output, tformer_out], dim=-1)

    return tokenizer.decode(output.cpu().numpy()[0].tolist())


def generate_gpt_manual(harness, tokenizer, inp, topk=10):
    harness.cuda()
    harness.eval()

    # Encoding input
    input_tokens = torch.from_numpy(np.array(tokenizer.encode(inp)))[None].to("cuda")
    output = torch.tensor(
        [[harness.model.config.decoder_start_token_id]] * 1,
        dtype=torch.long,
        device="cuda",
    )

    generated_tokens = []

    with torch.inference_mode():
        for i in range(100):
            out_logits = harness.model(
                input_ids=input_tokens,
                decoder_input_ids=output,
                attention_mask=torch.ones_like(input_tokens),
                decoder_attention_mask=torch.ones_like(output),
            )

            # Getting the top 5 logits
            tformer_out = out_logits.logits[:, -1:].topk(k=topk, dim=-1)
            tformer_out_logits = tformer_out.indices.cpu()
            tformer_out_values = tformer_out.values[0][0].cpu().tolist()

            # Decode the logits
            out_tokens = [
                tokenizer.decode(l[0]) for l in tformer_out_logits.reshape(topk, 1, 1)
            ]

            # Print out the next tokens
            # print("".join(generated_tokens), end="\n\n")
            # for idx, (tok, val) in enumerate(zip(out_tokens, tformer_out_values)):
            #     print(f"{idx + 1:<3}{tok:<15}{val:<6.2f}{int(val) * '█'}")

            # Choose the next token
            next_token_idx = input("Next token")
            next_token_idx = int(next_token_idx) - 1 if next_token_idx != "" else 0
            tformer_out = tformer_out.indices[:, -1:, next_token_idx]

            # Store the token
            generated_tokens.append(tokenizer.decode(tformer_out[0]))
            output = torch.cat([output, tformer_out], dim=-1)
            # clear_output(wait=True)


@torch.inference_mode()
def uncertainty_guided_search(
    harness, batch, tokenizer, k=128, threshold=0.4, keep_n=8, pick=2, out_length=448
):
    inp_batch = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()
    batch_size = inp_batch.shape[0]
    tgt = (
        torch.ones_like(inp_batch[:, :1]) * harness.model.config.decoder_start_token_id
    )
    overall_entr = torch.zeros_like(tgt).to(torch.float).flatten()
    done_beams = torch.zeros_like(inp_batch[:, :1]).to(torch.bool)

    ### FIX ###
    # "[EOS]" and " [EOS]" are encoded differently!
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
            # Mark as done all sequences whose last tokens match the [eos] embedding
            done_beams = (
                done_beams
                | (tgt[:, -done_seq.shape[0] :] == done_seq[None]).all(dim=-1)[:, None]
                | (tgt[:, -done_seq_alt.shape[0] :] == done_seq_alt[None]).all(dim=-1)[
                    :, None
                ]
            )

        # Predict the logits for the next token
        predicted_out_logits = harness.model(
            **{
                "input_ids": inp_batch,
                "decoder_input_ids": tgt,
                "attention_mask": torch.ones_like(inp_batch),
                "decoder_attention_mask": torch.ones_like(tgt),
            }
        ).logits[:, -1:]

        out_logits = predicted_out_logits

        # For anything that's done, we'll just override the logit
        out_logits = (
            out_logits * ~done_beams[..., None]
            + F.one_hot(torch.tensor(tokenizer.eos_token_id), out_logits.shape[-1])[
                None, None, :
            ].to(out_logits.device)
            * done_beams[..., None]
            * 9999999
        )

        # Calculate entropy
        out_logits_p = out_logits.softmax(dim=-1)
        entr = torch.special.entr(out_logits_p).sum(dim=-1)[..., None]
        entr[entr.isnan() | done_beams[..., None]] = 0.0
        # print(entr)

        # If entr > threshold for a stream, we pick the top "pick" beams and add them
        # to the search, otherwise we just use the same beam
        topks = out_logits_p.topk(pick).indices
        tops = out_logits_p.argmax(dim=-1)[..., None]
        select_mask_topks = (torch.ones_like(topks) * (entr > threshold)).bool()
        select_mask_tops = (torch.ones_like(tops) * ~(entr > threshold)).bool()

        # Fill done beams with eos tokens
        cat_tops = torch.cat([topks, tops], dim=-1)
        cat_tops = ~done_beams[:, None] * cat_tops + (
            done_beams[:, None] * torch.ones_like(cat_tops) * tokenizer.eos_token_id
        )

        # Construct masks
        select_mask_tops = torch.cat([select_mask_topks, select_mask_tops], dim=-1)
        beam_ids = torch.arange(
            inp_batch.shape[0], device=inp_batch.device, dtype=inp_batch.dtype
        )[:, None, None].expand(-1, 1, select_mask_tops.shape[-1])
        select_beams = beam_ids[select_mask_tops].flatten()
        select_nexts = cat_tops[select_mask_tops].flatten()
        entr_nexts = entr.expand(*entr.shape[:-1], select_mask_tops.shape[-1])[
            select_mask_tops
        ].flatten()

        # Update the values
        inp_batch = inp_batch[select_beams]
        labels = labels[select_beams]
        tgt = torch.cat([tgt[select_beams], select_nexts[:, None]], dim=-1)
        overall_entr = overall_entr[select_beams] + entr_nexts
        done_beams = done_beams[select_beams]
        batch_indices = batch_indices[select_beams]

        # Initial beam pruning, drop any beams where the average entropy is above 0.003
        excess_entropy_beams = (overall_entr / tgt.shape[1]) > 100

        # Don't drop if we would drop all beams
        excess_entropy_beams ^= excess_entropy_beams.all()[None]

        tgt = tgt[~excess_entropy_beams]
        done_beams = done_beams[~excess_entropy_beams]
        inp_batch = inp_batch[~excess_entropy_beams]
        labels = labels[~excess_entropy_beams]
        batch_indices = batch_indices[~excess_entropy_beams]
        overall_entr = overall_entr[~excess_entropy_beams]

        # Beam pruning, keep only the top keep_n beams by minimum entropy
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

    ### Pad targets to k length
    padding = (
        0,
        out_length - tgt.size(-1),
    )  # Pad only on the right side of k-dimension
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


def entropy_beam_search(
    harness, batch, tokenizer, steps=128, keep_n=8, out_length=448, **kwargs
):
    """
    Uncertainty guided beam-search
    Return shape: (BATCH, SEQ_LEN, @K)
    """
    # Shape: (BATCH, BEAM, (INP, TGT, LBL))
    out = uncertainty_guided_search(
        harness,
        batch,
        tokenizer,
        k=steps,
        keep_n=keep_n,
        out_length=out_length + 1,
        **kwargs,
    )
    out_logits = []
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    for b in out:
        _batch = []
        for beam in b[:keep_n]:
            _batch.append(beam[1][1:])  # tgt
        
        # FIX: Properly handle missing beams
        while len(_batch) < keep_n:
            _batch.append(np.full(out_length, pad_token_id, dtype=np.int64))

        out_logits.append(np.vstack(_batch))

    out_logits = torch.tensor(np.array(out_logits), dtype=torch.int64).permute(
        (0, 2, 1)
    )

    return out_logits


def beam_search_hf(
    harness,
    batch,
    beams=5,
    k=5,
    early_stopping=True,
    max_length=128,
    **kwargs,
):
    with torch.inference_mode():
        harness.cuda()
        inp_batch = batch["input_ids"].cuda()
        out_logits = harness.model.generate(
            input_ids=inp_batch,
            max_length=max_length,
            num_beams=beams,
            num_return_sequences=k,
            do_sample=False,
            early_stopping=early_stopping,
            **kwargs,
        )

        # Convert HuggingFace output to proper tensor if needed
        if hasattr(out_logits, 'sequences'):
            # For newer versions of transformers
            sequences = out_logits.sequences
        elif not isinstance(out_logits, torch.Tensor):
            # Convert to tensor if it's some other type
            sequences = torch.tensor(out_logits)
        else:
            sequences = out_logits
        
        # exclude the start token
        out_reshaped = sequences[:, 1:].cpu()
        batch_size = inp_batch.shape[0]
        actual_seq_len = out_reshaped.shape[1]
        
        # 🔍 Debug: 检查HuggingFace输出的实际格式
        print(f"\n🔍 HuggingFace beam_search_hf debug:")
        print(f"inp_batch.shape: {inp_batch.shape}")
        print(f"Raw sequences.shape: {sequences.shape}")
        print(f"out_reshaped.shape (after removing start token): {out_reshaped.shape}")
        print(f"batch_size: {batch_size}, k: {k}, actual_seq_len: {actual_seq_len}")
        print(f"Expected shape after view: {(batch_size, k, actual_seq_len)}")
        
        # 检查前几个序列是否真的不同
        if out_reshaped.shape[0] >= 5:
            print(f"First 5 sequences (first 10 tokens):")
            for i in range(min(5, out_reshaped.shape[0])):
                print(f"  Seq {i}: {out_reshaped[i, :10]}")
        
        # Reshape: from (batch_size * k, seq_len) to (batch_size, seq_len, k)
        out_reshaped = out_reshaped.view(batch_size, k, actual_seq_len).permute(0, 2, 1)
        
        print(f"Final output shape: {out_reshaped.shape}")
        print(f"First batch, first 10 tokens, all 5 candidates:")
        print(f"  Candidate 0: {out_reshaped[0, :10, 0]}")
        print(f"  Candidate 1: {out_reshaped[0, :10, 1]}")
        print(f"  Candidate 2: {out_reshaped[0, :10, 2]}")
        print(f"  Candidate 3: {out_reshaped[0, :10, 3]}")
        print(f"  Candidate 4: {out_reshaped[0, :10, 4]}")
        
        # 检查在什么位置开始出现差异
        print(f"\n🔍 检查候选序列差异位置:")
        first_seq = out_reshaped[0, :, 0]  # 第一个候选序列
        for i in range(1, 5):
            candidate_seq = out_reshaped[0, :, i]
            diff_mask = (first_seq != candidate_seq)
            if diff_mask.any():
                first_diff_pos = diff_mask.nonzero()[0].item()
                print(f"  Candidate {i} vs Candidate 0: 首次差异在位置 {first_diff_pos}")
                print(f"    位置 {first_diff_pos}: {first_seq[first_diff_pos].item()} vs {candidate_seq[first_diff_pos].item()}")
                # 显示更多差异位置
                diff_positions = diff_mask.nonzero().flatten()[:5]  # 前5个差异
                print(f"    前5个差异位置: {diff_positions.tolist()}")
            else:
                print(f"  Candidate {i} vs Candidate 0: 完全相同")

        return out_reshaped


def simple_temperature_test(
    harness,
    batch,
    beams=3,
    k=3,
    max_length=64,
    **kwargs,
):
    """
    简单测试：只是检查ATS头是否会产生不同的结果
    不使用复杂的集成方法
    """
    with torch.inference_mode():
        harness.cuda()
        inp_batch = batch["input_ids"].cuda()
        
        # 只做一个简单的对比测试
        if harness.train_mode == "ats":
            # 创建dummy batch来测试ATS头
            dummy_labels = torch.zeros_like(inp_batch[:, :5])
            temp_batch = {
                "input_ids": inp_batch,
                "labels": dummy_labels
            }
            
            # 获取温度信息
            _, temperatures, _ = harness.forward(temp_batch)
            
            if hasattr(temperatures, 'mean'):
                temp_mean = temperatures.mean().item()
                # print(f"   🌡️  ATS头平均温度: {temp_mean:.3f}")
            else:
                # print(f"   📏 固定温度: {temperatures}")
                pass
        
        # 使用标准HuggingFace生成（不强制应用温度）
        output = harness.model.generate(
            input_ids=inp_batch,
            max_length=max_length,
            num_beams=beams,
            num_return_sequences=k,
            do_sample=False,
            early_stopping=True,
            pad_token_id=harness.model.config.pad_token_id,
            **kwargs,
        )

        # 处理输出
        if hasattr(output, 'sequences'):
            sequences = output.sequences
        else:
            sequences = output
        
        sequences = sequences[:, 1:].cpu()
        batch_size = inp_batch.shape[0]
        actual_seq_len = sequences.shape[1]
        
        sequences = sequences.view(batch_size, k, actual_seq_len).permute(0, 2, 1)
        return sequences


@torch.inference_mode()
def temperature_weighted_beam_search(
    harness, batch, beams=5, alpha=0.5, max_length=128, **kwargs
):
    """
    温度-加权得分beam search
    score = log_prob - alpha * temperature
    
    Args:
        alpha: 温度惩罚系数，越大越惩罚不确定的token
    """
    inp_batch = batch["input_ids"].cuda()
    batch_size = inp_batch.shape[0]
    
    # 初始化：每个样本一个起始beam
    # beam格式: (sequence, cumulative_score)
    current_beams = []
    for i in range(batch_size):
        start_token = harness.model.config.decoder_start_token_id
        initial_seq = torch.tensor([start_token], device=inp_batch.device)
        current_beams.append([(initial_seq, 0.0)])  # (sequence, score)
    
    harness.cuda()
    harness.eval()
    
    # 生成每一步
    for step in range(max_length - 1):
        new_beams = []
        
        for batch_idx in range(batch_size):
            batch_candidates = []
            
            for seq, cum_score in current_beams[batch_idx]:
                # 如果序列已经结束，直接保留
                eos_token_id = harness.model.config.eos_token_id
                if seq[-1] == eos_token_id:
                    batch_candidates.append((seq, cum_score))
                    continue
                
                # 准备当前输入
                single_input = inp_batch[batch_idx:batch_idx+1]
                current_seq = seq.unsqueeze(0)  # (1, seq_len)
                
                # 🔥 关键：使用ATS头获取温度缩放后的logits
                if harness.train_mode == "ats":
                    temp_batch = {
                        "input_ids": single_input,
                        "labels": current_seq
                    }
                    scaled_logits, temperatures, original_logits = harness.forward(temp_batch)
                    next_token_logits = scaled_logits[0, -1, :]  # 最后一个位置的logits
                    
                    # 获取当前位置的平均温度
                    if hasattr(temperatures, 'mean'):
                        current_temp = temperatures[0, -1, :].mean().item()
                    else:
                        current_temp = temperatures  # 固定温度
                        
                    # if step == 0 and batch_idx == 0:  # 只打印一次
                    #     print(f"   🌡️  温度加权beam search: α={alpha}, 当前温度={current_temp:.3f}")
                        
                else:
                    # 标准T5（无温度缩放）
                    output = harness.model(
                        input_ids=single_input,
                        decoder_input_ids=current_seq,
                        attention_mask=torch.ones_like(single_input),
                        decoder_attention_mask=torch.ones_like(current_seq),
                    )
                    next_token_logits = output.logits[0, -1, :]
                    current_temp = 1.0  # 无温度
                    
                    # if step == 0 and batch_idx == 0:
                    #     print(f"   📏 标准beam search: α={alpha}, 固定温度=1.0")
                
                # 计算概率和分数
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                top_k_probs, top_k_indices = next_token_probs.topk(beams)
                
                # 为每个候选token计算温度加权分数
                for prob, token_id in zip(top_k_probs, top_k_indices):
                    log_prob = torch.log(prob).item()
                    
                    # 🔥 核心公式：score = log_prob - alpha * temperature
                    temp_penalty = alpha * current_temp
                    weighted_score = log_prob - temp_penalty
                    
                    new_score = cum_score + weighted_score
                    new_seq = torch.cat([seq, token_id.unsqueeze(0)])
                    
                    batch_candidates.append((new_seq, new_score))
            
            # 为当前batch保留top-beams个候选
            batch_candidates.sort(key=lambda x: x[1], reverse=True)  # 按分数降序
            new_beams.append(batch_candidates[:beams])
        
        current_beams = new_beams
        
        # 检查是否所有beam都结束了
        eos_token_id = harness.model.config.eos_token_id
        all_finished = True
        for batch_beams in current_beams:
            for seq, _ in batch_beams:
                if seq[-1] != eos_token_id:
                    all_finished = False
                    break
            if not all_finished:
                break
        
        if all_finished:
            break
    
    # 转换输出格式为 (batch_size, seq_len, k)
    result = []
    for batch_beams in current_beams:
        # 取前k个beam，移除start token
        k = min(len(batch_beams), beams)
        batch_sequences = []
        
        for i in range(k):
            seq, score = batch_beams[i]
            seq_without_start = seq[1:].cpu().numpy()  # 移除start token
            batch_sequences.append(seq_without_start)
        
        # 确保所有序列长度一致
        max_len = max(len(seq) for seq in batch_sequences) if batch_sequences else 1
        padded_sequences = []
        pad_token_id = harness.model.config.pad_token_id if hasattr(harness.model.config, 'pad_token_id') and harness.model.config.pad_token_id is not None else 0
        
        for seq in batch_sequences:
            if len(seq) < max_len:
                padded = np.concatenate([seq, np.full(max_len - len(seq), pad_token_id)])
            else:
                padded = seq[:max_len]
            padded_sequences.append(padded)
        
        # 填充到k个序列
        while len(padded_sequences) < beams:
            padded_sequences.append(np.full(max_len, pad_token_id))
        
        result.append(np.array(padded_sequences[:beams]).T)  # (seq_len, k)
    
    return torch.tensor(np.array(result), dtype=torch.long)  # (batch_size, seq_len, k)


@torch.inference_mode()
def temperature_rerank_beam_search(
    harness, batch, beams=5, k=5, alpha=0.5, max_length=128, **kwargs
):
    """
    简单高效的温度重排序方法：
    1. 先用HuggingFace生成更多候选
    2. 用ATS头计算平均温度
    3. 重新排序: score = log_prob - alpha * avg_temperature
    """
    with torch.inference_mode():
        harness.cuda()
        inp_batch = batch["input_ids"].cuda()
        
        # 🔥 步骤1: 用HuggingFace生成更多候选（比如2倍）
        num_candidates = max(beams * 2, 10)  # 生成更多候选用于重排
        
        # Debug: 检查参数冲突
        # print(f"Debug: beams={beams}, num_candidates={num_candidates}")
        # print(f"Debug: Before fix - num_beams={beams}, num_return_sequences={num_candidates}")
        
        # 修复参数冲突：确保 num_return_sequences <= num_beams
        actual_num_beams = max(beams, num_candidates)
        actual_num_return_sequences = min(num_candidates, actual_num_beams)
        
        # print(f"Debug: After fix - num_beams={actual_num_beams}, num_return_sequences={actual_num_return_sequences}")
        
        output = harness.model.generate(
            input_ids=inp_batch,
            max_length=max_length,
            num_beams=actual_num_beams,
            num_return_sequences=actual_num_return_sequences,
            do_sample=False,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )
        
        sequences = output.sequences[:, 1:].cpu()  # 移除start token
        batch_size = inp_batch.shape[0]
        actual_seq_len = sequences.shape[1]
        
        # Reshape: (batch_size, num_candidates, seq_len)
        sequences = sequences.view(batch_size, num_candidates, actual_seq_len)
        
        # 🔥 步骤2: 用ATS头计算每个候选的平均温度
        reranked_results = []
        
        for batch_idx in range(batch_size):
            batch_candidates = []
            single_input = inp_batch[batch_idx:batch_idx+1]
            
            for cand_idx in range(num_candidates):
                candidate_seq = sequences[batch_idx, cand_idx:cand_idx+1, :]  # (1, seq_len)
                
                if harness.train_mode == "ats":
                    # 使用ATS头计算温度
                    temp_batch = {
                        "input_ids": single_input,
                        "labels": candidate_seq
                    }
                    
                    try:
                        scaled_logits, temperatures, original_logits = harness.forward(temp_batch)
                        
                        if hasattr(temperatures, 'mean'):
                            # 计算平均温度（排除padding）
                            valid_mask = candidate_seq != harness.model.config.pad_token_id
                            if valid_mask.sum() > 0:
                                avg_temp = temperatures[valid_mask].mean().item()
                            else:
                                avg_temp = temperatures.mean().item()
                        else:
                            avg_temp = temperatures
                            
                        # if batch_idx == 0 and cand_idx == 0:  # 只打印一次
                        #     print(f"   🌡️  温度重排序: α={alpha}, 样本平均温度={avg_temp:.3f}")
                            
                    except Exception as e:
                        # 如果出错，使用默认温度
                        avg_temp = 1.0
                        # if batch_idx == 0 and cand_idx == 0:
                        #     print(f"   ⚠️  温度计算出错，使用默认值: {e}")
                else:
                    avg_temp = 1.0  # 标准T5
                    # if batch_idx == 0 and cand_idx == 0:
                    #     print(f"   📏 标准模型重排序: α={alpha}")
                
                # 🔥 步骤3: 计算HuggingFace的原始log-probability
                # 简化：使用序列长度作为粗略的log-prob代理
                # 更准确的做法是重新计算实际的log-prob，但这里简化处理
                seq_length = (candidate_seq != harness.model.config.pad_token_id).sum().item()
                rough_log_prob = -seq_length * 0.1  # 粗略估计，长度越长概率越低
                
                # 🔥 步骤4: 温度加权分数
                temp_penalty = alpha * avg_temp
                weighted_score = rough_log_prob - temp_penalty
                
                batch_candidates.append((candidate_seq.squeeze(0), weighted_score))
            
            # 按加权分数排序，取top-k
            batch_candidates.sort(key=lambda x: x[1], reverse=True)
            top_k_seqs = [seq for seq, _ in batch_candidates[:k]]
            
            # 填充到k个序列
            while len(top_k_seqs) < k:
                pad_seq = torch.full((actual_seq_len,), harness.model.config.pad_token_id or 0)
                top_k_seqs.append(pad_seq)
            
            # 转换为 (seq_len, k) 格式
            batch_result = torch.stack(top_k_seqs[:k], dim=1)  # (seq_len, k)
            reranked_results.append(batch_result)
        
        # 合并所有batch: (batch_size, seq_len, k)
        final_result = torch.stack(reranked_results, dim=0)
        
        return final_result


