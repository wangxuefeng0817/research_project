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
    print(tokenizer.batch_decode(input_tokens))
    print(output)

    with torch.inference_mode():
        for i in range(max_steps):
            out_logits = harness.model(
                input_ids=input_tokens,
                decoder_input_ids=output,
                attention_mask=torch.ones_like(input_tokens),
                decoder_attention_mask=torch.ones_like(output),
            )
            import pdb

            pdb.set_trace()
            tformer_out = out_logits.logits[:, -1:].argmax(dim=-1)
            print(tformer_out)
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
            print("".join(generated_tokens), end="\n\n")
            for idx, (tok, val) in enumerate(zip(out_tokens, tformer_out_values)):
                print(f"{idx + 1:<3}{tok:<15}{val:<6.2f}{int(val) * '█'}")

            # Choose the next token
            next_token_idx = input("Next token")
            next_token_idx = int(next_token_idx) - 1 if next_token_idx != "" else 0
            tformer_out = tformer_out.indices[:, -1:, next_token_idx]

            # Store the token
            generated_tokens.append(tokenizer.decode(tformer_out[0]))
            output = torch.cat([output, tformer_out], dim=-1)
            clear_output(wait=True)


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
    for b in out:
        _batch = []
        for beam in b[:keep_n]:
            _batch.append(beam[1][1:])  # tgt
        if len(_batch) < keep_n:
            # Zero pad missing beams
            pad_missing_beams = np.zeros((keep_n - len(_batch), out_length))
            _batch.append(pad_missing_beams)

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
    max_beam_lenth=128,
    out_length=448,
    early_stopping=True,
    **kwargs,
):
    with torch.inference_mode():
        harness.cuda()
        inp_batch = batch["input_ids"].cuda()
        out_logits = harness.model.generate(
            input_ids=inp_batch,
            max_length=max_beam_lenth + 1,
            num_beams=beams,
            num_return_sequences=k,
            do_sample=False,
            early_stopping=early_stopping,
            **kwargs,
        )

        # exclude the start token
        out_reshaped = out_logits[:, 1:].cpu()
        # reshape to (bath, seq_length, k)
        out_reshaped = out_reshaped.view(-1, k, max_beam_lenth).permute(0, 2, 1)

        # MODIFICATION: Pad targets to k length
        padded_tensor = F.pad(
            out_reshaped, (0, 0, 0, out_length - out_reshaped.size(1)), value=2
        )

        return padded_tensor