import sys
import time
import torch
import numpy as np
import torch.nn.functional as F

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

@torch.inference_mode()
def fixed_temperature_guided_search(
    harness, batch, tokenizer, k=128, temp_threshold=1.0, keep_n=8, pick=3, out_length=448
):
    """
    Fixed version of uncertainty guided search using ATS temperatures.
    
    Key fixes:
    1. Proper tensor shape handling
    2. More reasonable temperature threshold (1.0 instead of 0.4)
    3. Better beam pruning strategy
    4. Robust error handling
    """
    inp_batch = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()
    batch_size = inp_batch.shape[0]
    
    # Initialize with decoder start token
    tgt = torch.ones((batch_size, 1), dtype=torch.long, device=inp_batch.device) * harness.model.config.decoder_start_token_id
    
    # Track beam scores (log probabilities)
    beam_scores = torch.zeros(batch_size, device=inp_batch.device)
    
    # Track which beams are done
    done_beams = torch.zeros(batch_size, dtype=torch.bool, device=inp_batch.device)
    batch_indices = torch.arange(batch_size, device=inp_batch.device)

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    harness.eval()

    for step in range(k):
        if done_beams.all():
            break

        # Check for EOS tokens
        if step > 0:
            is_done = (tgt[:, -1] == eos_token_id)
            done_beams = done_beams | is_done

        # Get model output with ATS temperatures
        try:
            model_output = harness.forward({
                "input_ids": inp_batch,
                "decoder_input_ids": tgt,
                "attention_mask": torch.ones_like(inp_batch),
                "decoder_attention_mask": torch.ones_like(tgt),
                "labels": labels  # Need labels for ATS forward pass
            })
            
            if len(model_output) == 3:
                scaled_logits, temperatures, original_logits = model_output
                # Use original logits for more stable generation
                current_logits = original_logits[:, -1, :]  # Last token logits
                current_temps = temperatures[:, -1].squeeze(-1)  # Last token temperatures
            else:
                # Fallback if ATS not available
                current_logits = harness.model(
                    input_ids=inp_batch,
                    decoder_input_ids=tgt,
                    attention_mask=torch.ones_like(inp_batch),
                    decoder_attention_mask=torch.ones_like(tgt)
                ).logits[:, -1, :]
                current_temps = torch.ones(current_logits.shape[0], device=current_logits.device)
                
        except Exception as e:
            print(f"Warning: ATS forward failed, using base model: {e}")
            current_logits = harness.model(
                input_ids=inp_batch,
                decoder_input_ids=tgt,
                attention_mask=torch.ones_like(inp_batch),
                decoder_attention_mask=torch.ones_like(tgt)
            ).logits[:, -1, :]
            current_temps = torch.ones(current_logits.shape[0], device=current_logits.device)

        # Get log probabilities
        log_probs = F.log_softmax(current_logits, dim=-1)
        
        # Mask finished beams
        log_probs[done_beams] = -float('inf')
        log_probs[done_beams, pad_token_id] = 0.0  # Allow padding for finished beams

        # Decision: Use multiple candidates if temperature is high
        should_expand = (current_temps > temp_threshold) & (~done_beams)
        
        # For high-temperature tokens, consider top-k candidates
        # For low-temperature tokens, use greedy (top-1)
        
        all_candidates = []
        all_scores = []
        all_beam_indices = []
        
        # Process each beam
        for beam_idx in range(current_logits.shape[0]):
            if done_beams[beam_idx]:
                # Keep the current sequence for done beams
                all_candidates.append(pad_token_id)
                all_scores.append(beam_scores[beam_idx])
                all_beam_indices.append(beam_idx)
            elif should_expand[beam_idx]:
                # High temperature: consider multiple candidates
                top_scores, top_indices = torch.topk(log_probs[beam_idx], min(pick, log_probs.shape[-1]))
                for score, token_id in zip(top_scores, top_indices):
                    all_candidates.append(token_id.item())
                    all_scores.append(beam_scores[beam_idx] + score.item())
                    all_beam_indices.append(beam_idx)
            else:
                # Low temperature: greedy selection
                best_token = log_probs[beam_idx].argmax()
                best_score = log_probs[beam_idx].max()
                all_candidates.append(best_token.item())
                all_scores.append(beam_scores[beam_idx] + best_score.item())
                all_beam_indices.append(beam_idx)

        # Convert to tensors
        all_candidates = torch.tensor(all_candidates, device=inp_batch.device)
        all_scores = torch.tensor(all_scores, device=inp_batch.device)
        all_beam_indices = torch.tensor(all_beam_indices, device=inp_batch.device)

        # Select top keep_n*batch_size candidates globally, but ensure representation from each batch
        new_tgt_list = []
        new_scores_list = []
        new_batch_indices_list = []

        for batch_id in range(batch_size):
            # Find candidates for this batch item
            batch_mask = (batch_indices[all_beam_indices] == batch_id)
            if not batch_mask.any():
                continue
                
            batch_candidates = all_candidates[batch_mask]
            batch_scores = all_scores[batch_mask]
            batch_beam_indices = all_beam_indices[batch_mask]
            
            # Select top keep_n for this batch
            if len(batch_scores) > keep_n:
                top_indices = batch_scores.topk(keep_n).indices
                selected_candidates = batch_candidates[top_indices]
                selected_scores = batch_scores[top_indices]
                selected_beam_indices = batch_beam_indices[top_indices]
            else:
                selected_candidates = batch_candidates
                selected_scores = batch_scores
                selected_beam_indices = batch_beam_indices

            # Extend sequences
            for i in range(len(selected_candidates)):
                beam_id = selected_beam_indices[i]
                new_seq = torch.cat([tgt[beam_id:beam_id+1], selected_candidates[i:i+1].unsqueeze(0)], dim=1)
                new_tgt_list.append(new_seq)
                new_scores_list.append(selected_scores[i])
                new_batch_indices_list.append(batch_id)

        if not new_tgt_list:
            break

        # Update state
        tgt = torch.cat(new_tgt_list, dim=0)
        beam_scores = torch.tensor(new_scores_list, device=inp_batch.device)
        batch_indices = torch.tensor(new_batch_indices_list, device=inp_batch.device)
        
        # Update other tensors
        inp_batch = batch["input_ids"].cuda()[batch_indices]
        labels = batch["labels"].cuda()[batch_indices]
        done_beams = done_beams[batch_indices]

    # Finalize results
    final_results = []
    for batch_id in range(batch_size):
        batch_mask = batch_indices == batch_id
        if batch_mask.any():
            batch_seqs = tgt[batch_mask]
            batch_scores = beam_scores[batch_mask]
            
            # Sort by score
            sorted_indices = batch_scores.argsort(descending=True)
            sorted_seqs = batch_seqs[sorted_indices]
            
            # Pad to required dimensions
            padded_seqs = torch.full((keep_n, out_length), pad_token_id, 
                                   dtype=torch.long, device=tgt.device)
            
            for i in range(min(keep_n, len(sorted_seqs))):
                seq = sorted_seqs[i]
                seq_len = min(out_length, seq.shape[0])
                padded_seqs[i, :seq_len] = seq[:seq_len]
            
            final_results.append(padded_seqs)
        else:
            # Empty result for this batch
            padded_seqs = torch.full((keep_n, out_length), pad_token_id,
                                   dtype=torch.long, device=inp_batch.device)
            final_results.append(padded_seqs)

    result_tensor = torch.stack(final_results)
    # Permute to match expected output format (batch, seq, beam)
    return result_tensor.permute(0, 2, 1)


@torch.inference_mode()
def fixed_temperature_rerank_search(
    harness,
    batch,
    tokenizer,
    beams=8,
    max_length=128,
    alpha=0.3,  # Reduced from 0.5 to be less aggressive
    **kwargs,
):
    """
    Fixed version of temperature rerank search with proper error handling.
    """
    harness.eval()
    if torch.cuda.is_available():
        harness.cuda()

    inp_batch = batch["input_ids"]
    labels = batch["labels"]
    if torch.cuda.is_available():
        inp_batch = inp_batch.to(harness.device)
        labels = labels.to(harness.device)

    # 1. Generate candidates using standard beam search
    try:
        beam_output = harness.model.generate(
            inp_batch,
            num_beams=beams,
            max_length=max_length,
            num_return_sequences=beams,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
            **kwargs,
        )
        sequences = beam_output.sequences
        sequence_scores = beam_output.sequences_scores
    except Exception as e:
        print(f"Warning: Beam search failed: {e}")
        # Fallback to simpler generation
        sequences = harness.model.generate(
            inp_batch,
            num_beams=beams,
            max_length=max_length,
            num_return_sequences=beams,
            **kwargs,
        )
        # Create dummy scores
        sequence_scores = torch.zeros(sequences.shape[0], device=sequences.device)
    
    batch_size = inp_batch.shape[0]
    
    # Handle case where sequences might be shorter than expected
    if len(sequences.shape) == 2:
        sequences_reshaped = sequences.view(batch_size, beams, -1)
    else:
        sequences_reshaped = sequences
        
    if len(sequence_scores.shape) == 1:
        scores_reshaped = sequence_scores.view(batch_size, beams)
    else:
        scores_reshaped = sequence_scores

    all_reranked_sequences = []

    for i in range(batch_size):
        try:
            # 2. Calculate temperatures for each candidate
            current_input = inp_batch[i:i+1].expand(beams, -1)
            current_labels = labels[i:i+1].expand(beams, -1)
            current_sequences = sequences_reshaped[i]
            
            # Get temperatures - process one by one to avoid shape issues
            avg_temperatures = []
            for j in range(beams):
                try:
                    single_batch = {
                        "input_ids": current_input[j:j+1],
                        "decoder_input_ids": current_sequences[j:j+1],
                        "attention_mask": torch.ones_like(current_input[j:j+1]),
                        "decoder_attention_mask": torch.ones_like(current_sequences[j:j+1]),
                        "labels": current_sequences[j:j+1]  # Use the sequence itself as label
                    }
                    
                    model_output = harness.forward(single_batch)
                    if len(model_output) == 3:
                        _, temperatures, _ = model_output
                        
                        # Calculate average temperature for non-pad tokens
                        non_pad_mask = current_sequences[j] != tokenizer.pad_token_id
                        if non_pad_mask.any() and temperatures.shape[1] > 0:
                            # Take average over valid positions
                            valid_temps = temperatures[0, :non_pad_mask.sum()]
                            avg_temp = valid_temps.mean().item()
                        else:
                            avg_temp = 1.0  # Default temperature
                    else:
                        avg_temp = 1.0  # Fallback if ATS not available
                        
                    avg_temperatures.append(avg_temp)
                except Exception as e:
                    print(f"Warning: Temperature calculation failed for beam {j}: {e}")
                    avg_temperatures.append(1.0)  # Default temperature
            
            # 3. Re-rank based on quality score
            new_scores = []
            for j in range(beams):
                seq_len = (current_sequences[j] != tokenizer.pad_token_id).sum().item()
                if seq_len > 0:
                    # Normalize by sequence length
                    normalized_log_prob = scores_reshaped[i, j].item() / seq_len
                    # Temperature penalty: higher temperature = lower score
                    temp_penalty = alpha * (avg_temperatures[j] - 1.0)  # Penalty for temp > 1.0
                    new_score = normalized_log_prob - temp_penalty
                else:
                    new_score = -float('inf')
                new_scores.append(new_score)
            
            # Get reranking order (best score first)
            reranked_indices = sorted(range(len(new_scores)), key=lambda x: new_scores[x], reverse=True)
            
            # Reorder sequences
            reranked_sequences = current_sequences[reranked_indices]
            all_reranked_sequences.append(reranked_sequences)
            
        except Exception as e:
            print(f"Warning: Reranking failed for batch {i}: {e}")
            # Fallback: use original order
            all_reranked_sequences.append(sequences_reshaped[i])

    return torch.stack(all_reranked_sequences)


def beam_search_hf(
    harness,
    batch,
    tokenizer,
    beams=5,
    max_length=128,
    early_stopping=True,
    **kwargs,
):
    """Standard beam search using huggingface's built-in generate"""
    harness.eval()
    if torch.cuda.is_available():
        harness.cuda()

    inp_batch = batch["input_ids"]
    if torch.cuda.is_available():
        inp_batch = inp_batch.to(harness.device)

    try:
        output = harness.model.generate(
            inp_batch,
            num_beams=beams,
            max_length=max_length,
            early_stopping=early_stopping,
            num_return_sequences=beams,
            do_sample=False,
            **kwargs,
        )
        
        # Handle different output types from HuggingFace generate
        if hasattr(output, 'sequences'):
            sequences = output.sequences
        else:
            sequences = output
        
        # Reshape to (batch_size, beams, seq_len)
        batch_size = inp_batch.shape[0]
        return sequences.view(batch_size, beams, -1)
        
    except Exception as e:
        print(f"Error in beam search: {e}")
        # Ultra-simple fallback: greedy generation
        output = harness.model.generate(
            inp_batch,
            max_length=max_length,
            do_sample=False,
            **kwargs,
        )
        batch_size = inp_batch.shape[0]
        # Expand single sequence to multiple beams
        expanded = output.unsqueeze(1).repeat(1, beams, 1)
        return expanded


# Wrapper functions for compatibility
def fixed_temperature_rerank_beam_search(
    harness, batch, tokenizer, steps=128, keep_n=8, out_length=448, **kwargs
):
    """Wrapper for fixed temperature rerank search."""
    try:
        reranked_sequences = fixed_temperature_rerank_search(
            harness=harness,
            batch=batch,
            tokenizer=tokenizer,
            beams=keep_n,
            max_length=out_length,
            **kwargs
        )
        # Permute from (batch, beam, seq) to (batch, seq, beam)
        return reranked_sequences.permute(0, 2, 1)
    except Exception as e:
        print(f"Error in temperature rerank: {e}")
        # Fallback to standard beam search
        return beam_search_hf(harness, batch, tokenizer, beams=keep_n, max_length=out_length).permute(0, 2, 1)


def fixed_temperature_beam_search(
    harness, batch, tokenizer, steps=128, keep_n=8, out_length=448, **kwargs
):
    """Wrapper for fixed temperature guided search."""
    try:
        return fixed_temperature_guided_search(
            harness=harness,
            batch=batch,
            tokenizer=tokenizer,
            k=steps,
            keep_n=keep_n,
            out_length=out_length,
            **kwargs
        )
    except Exception as e:
        print(f"Error in temperature guided search: {e}")
        # Fallback to standard beam search
        return beam_search_hf(harness, batch, tokenizer, beams=keep_n, max_length=out_length).permute(0, 2, 1) 