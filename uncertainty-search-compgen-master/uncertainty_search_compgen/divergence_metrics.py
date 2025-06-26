import torch
import torch.nn.functional as F


def measure_teacher_student_model_divergence(harness, student_harness, batch):
    with torch.inference_mode():
        inp_batch = batch["input_ids"].cuda()
        outp_batch = batch["labels"].cuda()
        tgt = outp_batch

        harness.cuda()
        harness.eval()
        teacher_predicted_out_logits = harness.model(
            **{
                "input_ids": inp_batch,
                "labels": tgt,
            }
        ).logits.log_softmax(dim=-1)

        student_harness.cuda()
        student_harness.eval()
        student_predicted_out_logits = student_harness.model(
            **{
                "input_ids": inp_batch,
                "labels": tgt,
            }
        ).logits.log_softmax(dim=-1)

        return (
            F.kl_div(
                teacher_predicted_out_logits,
                student_predicted_out_logits,
                reduction="none",
                log_target=True,
            )
            .sum(dim=-1)
            .cpu()
            .numpy()
        )

        out_logits = predicted_out_logits
    return (
        torch.special.entr(F.softmax(out_logits, dim=-1))
        .sum(dim=-1)
        .detach()
        .cpu()
        .numpy()
    )


def measure_mutual_kl_causal_mask(harness, batch, nb=5):
    harness.cuda()
    inp_batch = batch["input_ids"].cuda()
    outp_batch = batch["labels"].cuda()
    tgt = outp_batch

    harness.train()
    out_logits = harness.model(
        **{
            "input_ids": torch.repeat_interleave(inp_batch, nb, dim=0),
            "labels": torch.repeat_interleave(tgt, nb, dim=0),
        }
    ).logits
    # (B x NB) x L x E => B x NB x L x E
    out_logits_p = out_logits.softmax(dim=-1).view(
        inp_batch.shape[0], nb, outp_batch.shape[1], out_logits.shape[2]
    )
    out_logits_logp = out_logits.log_softmax(dim=-1).view(
        inp_batch.shape[0], nb, outp_batch.shape[1], out_logits.shape[2]
    )

    # we want p (p log q - p log p).sum(dim=-1) => B x L x E x NB x 1 * (B x L x E x NB x 1 - B x L x E x 1 x NB) => B x L x E x NB x NB
    # out_logits_logq_minus_logp = (
    #    out_logits_p.permute(0, 2, 3, 1)[..., None] * (
    #        torch.log(out_logits_p.permute(0, 2, 3, 1)[..., None]) -
    #        torch.log(out_logits_p.permute(0, 2, 3, 1)[..., None, :])
    #    )
    # ).sum(dim=-3)
    out_logits_logq_minus_logp = (
        (
            out_logits_p.permute(0, 2, 3, 1)[..., None]
            * torch.log(out_logits_p.permute(0, 2, 3, 1)[..., None])
            - out_logits_p.permute(0, 2, 3, 1)[..., None]
            * torch.log(out_logits_p.permute(0, 2, 3, 1)[..., None, :])
        )
    ).sum(dim=-3)

    # B x NB x L x E => B x NB x L
    # out_logits_ent = (-out_logits_p * out_logits_logp).sum(dim=-1)

    # (B x NB x L x E => B x L x NB x E) x (B x NB x E x L => B x L x E x NB) => B x L x NB x NB - B x L x 1 x NB
    # out_logits_kls = torch.triu(
    #    (
    #        out_logits_p.permute(0, 2, 1, 3) * torch.log(out_logits_p.permute(0, 2, 3, 1) - out_logits_p.permute(0, 2, 1, 3))
    #    ),
    #    diagonal=1
    # )
    out_logits_kls = torch.triu(out_logits_logq_minus_logp, diagonal=1)
    return (
        (
            (out_logits_kls / ((nb * (nb - 1)) / 2)).sum(dim=-1).sum(dim=-1)
            * (tgt != harness.hparams.pad_token)
        )
        .detach()
        .cpu()
        .numpy()
    )


def measure_entropy(harness, batch):
    with torch.inference_mode():
        harness.cuda()
        inp_batch = batch["input_ids"].cuda()
        outp_batch = batch["labels"].cuda()
        tgt = outp_batch

        harness.eval()
        predicted_out_logits = harness.model(
            **{
                "input_ids": inp_batch,
                "labels": tgt,
            }
        ).logits

        if False:
            out_logits = torch.ones_like(predicted_out_logits) * float("-inf")
            out_logits.scatter_(
                -1,
                permitted_transition_indices[tgt[-1]][:, None],
                torch.gather(
                    predicted_out_logits,
                    -1,
                    permitted_transition_indices[tgt[-1]][:, None],
                ),
            )
        else:
            out_logits = predicted_out_logits
        return (
            torch.special.entr(F.softmax(out_logits, dim=-1))
            .sum(dim=-1)
            .detach()
            .cpu()
            .numpy()
        )


