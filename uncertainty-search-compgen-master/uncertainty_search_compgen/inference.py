import torch
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


def get_topk_outputs(harness, batch, k=5):
    with torch.no_grad():
    #with torch.inference_mode():
        harness.cuda()
        inp_batch = batch["input_ids"].cuda()
        outp_batch = batch["labels"].cuda()
        tgt = outp_batch

        harness.eval()
        predicted_out_logits = harness.model(
            **{"input_ids": inp_batch, "labels": tgt}
        ).logits

        if False:
            out_logits = mask_permitted_transition_indices(
                predicted_out_logits, tgt[-1], permitted_transition_indices
            )
        else:
            out_logits = predicted_out_logits

        return out_logits.topk(k, dim=-1).indices.cpu().numpy()


def generate_gpt(harness, inp):
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

    #with torch.inference_mode():
    with torch.no_grad():
        for i in trange(128):
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


def uncertainty_guided_search(
    harness, batch, tokenizer, k=128, threshold=0.4, keep_n=8, pick=2
):
    inp_batch = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()
    batch_size = inp_batch.shape[0]
    tgt = (
        torch.ones_like(inp_batch[:, :1]) * harness.model.config.decoder_start_token_id
    )
    overall_entr = torch.zeros_like(tgt).to(torch.float).flatten()
    done_beams = torch.zeros_like(inp_batch[:, :1]).to(torch.bool)
    done_seq = torch.tensor(
        tokenizer.encode("[eos]")[1:-1], device=tgt.device, dtype=tgt.dtype
    )
    batch_indices = torch.arange(done_beams.shape[0], device="cuda", dtype=torch.int)

    harness.cuda()
    harness.eval()

    #with torch.inference_mode():
    with torch.no_grad():
        for i in range(k):
            if tgt.shape[-1] >= done_seq.shape[0]:
                done_beams = (
                    done_beams
                    | (tgt[:, -done_seq.shape[0] :] == done_seq[None]).all(dim=-1)[
                        :, None
                    ]
                )

            predicted_out_logits = harness.model(
                **{
                    "input_ids": inp_batch,
                    "decoder_input_ids": tgt,
                    "attention_mask": torch.ones_like(inp_batch),
                    "decoder_attention_mask": torch.ones_like(tgt),
                }
            ).logits[:, -1:]
            # import pdb
            # pdb.set_trace()

            # We modify the logits to only allow transitions that we know about
            if False:
                out_logits = mask_permitted_transition_indices(
                    predicted_out_logits, tgt[-1], permitted_transition_indices
                )
                # print([idx2word[w] for w in other_out_logits[0, 0].topk(dim=-1, k=5).indices.cpu().numpy()])
                # print([idx2word[w] for w in predicted_out_logits[0, 0].topk(dim=-1, k=5).indices.cpu().numpy()])
            else:
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
            out_logits_p = out_logits.softmax(dim=-1)
            entr = torch.special.entr(out_logits_p).sum(dim=-1)[..., None]
            entr[entr.isnan() | done_beams[..., None]] = 0.0
            # print(entr)

            # If entr > 1 for a stream, we pick the top 3 things and add them
            # to the search, otherwise we just use the same beam
            topks = out_logits_p.topk(pick).indices
            tops = out_logits_p.argmax(dim=-1)[..., None]
            select_mask_topks = (torch.ones_like(topks) * (entr > threshold)).bool()
            select_mask_tops = (torch.ones_like(tops) * ~(entr > threshold)).bool()

            cat_tops = torch.cat([topks, tops], dim=-1)
            cat_tops = ~done_beams[:, None] * cat_tops + (
                done_beams[:, None] * torch.ones_like(cat_tops) * tokenizer.eos_token_id
            )
            select_mask_tops = torch.cat([select_mask_topks, select_mask_tops], dim=-1)
            beam_ids = torch.arange(
                inp_batch.shape[0], device=inp_batch.device, dtype=inp_batch.dtype
            )[:, None, None].expand(-1, 1, select_mask_tops.shape[-1])
            select_beams = beam_ids[select_mask_tops].flatten()
            select_nexts = cat_tops[select_mask_tops].flatten()
            entr_nexts = entr.expand(*entr.shape[:-1], select_mask_tops.shape[-1])[
                select_mask_tops
            ].flatten()

            inp_batch = inp_batch[select_beams]
            labels = labels[select_beams]
            tgt = torch.cat([tgt[select_beams], select_nexts[:, None]], dim=-1)
            overall_entr = overall_entr[select_beams] + entr_nexts
            done_beams = done_beams[select_beams]
            batch_indices = batch_indices[select_beams]

            if False:
                import pprint

                pprint.pprint(
                    list(
                        zip(
                            batch_indices[overall_entr.argsort()].cpu().numpy(),
                            map(
                                lambda x: x,
                                tokenizer.batch_decode(tgt[overall_entr.argsort()]),
                            ),
                            overall_entr[overall_entr.argsort()].cpu().numpy(),
                            overall_entr[overall_entr.argsort()].cpu().numpy()
                            / tgt.shape[1],
                        )
                    )
                )

            is_matching = (labels[:, : tgt.shape[-1] - 1] == tgt[:, 1:]).all(dim=-1)

            if False and not is_matching.any():
                import pprint

                pprint.pprint(
                    list(
                        zip(
                            batch_indices[overall_entr.argsort()].cpu().numpy(),
                            map(
                                lambda x: x,
                                tokenizer.batch_decode(tgt[overall_entr.argsort()]),
                            ),
                            overall_entr[overall_entr.argsort()].cpu().numpy(),
                            overall_entr[overall_entr.argsort()].cpu().numpy()
                            / tgt.shape[1],
                        )
                    )
                )
                import pdb

                pdb.set_trace()

            # Initial beam pruning, drop any beams where the average entropy is above
            # 0.003
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
            tgt = torch.cat(
                [
                    tgt[batch_indices == i][
                        (overall_entr[batch_indices == i].flatten()).argsort()[:keep_n]
                    ]
                    for i in range(batch_size)
                ],
                dim=0,
            )
            done_beams = torch.cat(
                [
                    done_beams[batch_indices == i][
                        (overall_entr[batch_indices == i].flatten()).argsort()[:keep_n]
                    ]
                    for i in range(batch_size)
                ],
                dim=0,
            )
            inp_batch = torch.cat(
                [
                    inp_batch[batch_indices == i][
                        (overall_entr[batch_indices == i].flatten()).argsort()[:keep_n]
                    ]
                    for i in range(batch_size)
                ],
                dim=0,
            )
            labels = torch.cat(
                [
                    labels[batch_indices == i][
                        (overall_entr[batch_indices == i].flatten()).argsort()[:keep_n]
                    ]
                    for i in range(batch_size)
                ],
                dim=0,
            )
            next_batch_indices = torch.cat(
                [
                    batch_indices[batch_indices == i][
                        (overall_entr[batch_indices == i].flatten()).argsort()[:keep_n]
                    ]
                    for i in range(batch_size)
                ],
                dim=0,
            )
            overall_entr = torch.cat(
                [
                    overall_entr[batch_indices == i][
                        (overall_entr[batch_indices == i].flatten()).argsort()[:keep_n]
                    ]
                    for i in range(batch_size)
                ],
                dim=0,
            )
            batch_indices = next_batch_indices

            if done_beams.all(dim=0)[0].item():
                break

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


