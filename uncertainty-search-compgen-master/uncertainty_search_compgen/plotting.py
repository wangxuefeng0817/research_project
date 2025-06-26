import matplotlib.pyplot as plt
import pandas as pd


def plot_batch_and_metric(tokenizer, batch, loss, idx2word, pad_token):
    fig, ax1 = plt.subplots(figsize=(18, 4))

    color = "tab:blue"
    ax1.set_ylabel("loss", color=color)  # we already handled the x-label with ax1
    ax1.plot(loss[0][batch["labels"][0] != pad_token], label="loss", color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    idx2word = {i: w for w, i in tokenizer.vocab.items()}

    decoded_tokens = batch["labels"][0][batch["labels"][0] != pad_token].tolist()

    ax1.set_xticks(range(len(decoded_tokens)))
    ax1.set_xticklabels([idx2word[i] for i in decoded_tokens], rotation=45, ha="right")
    ax1.set_title(tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def plot_batch_and_metric_pair(tokenizer, batch, kl, loss, idx2word, pad_token):
    fig, ax1 = plt.subplots(figsize=(36, 12))

    color = "tab:red"
    ax1.set_xlabel("token")
    ax1.set_ylabel("ent", color=color)
    ax1.plot(kl[0][batch["labels"][0] != pad_token], label="kl", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_title(tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True))

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel("loss", color=color)  # we already handled the x-label with ax1
    ax2.plot(loss[0][batch["labels"][0] != pad_token], label="loss", color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    idx2word = {i: w for w, i in tokenizer.vocab.items()}

    decoded_tokens = batch["labels"][0][batch["labels"][0] != pad_token].tolist()

    ax1.set_xticks(range(len(decoded_tokens)))
    ax1.set_xticklabels([idx2word[i] for i in decoded_tokens], rotation=45, ha="right")
    print("done")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def visualize_as_table(ent, kl, loss, student_loss, topks, batch, idx2word, pad_token):
    df = pd.DataFrame.from_dict(
        {
            "ent": ent[0][batch["labels"][0] != pad_token],
            "kl": kl[0][batch["labels"][0] != pad_token],
            "loss": loss[0][batch["labels"][0] != pad_token],
            "student_loss": student_loss[0][batch["labels"][0] != pad_token],
            "true": [
                idx2word[w]
                for w in batch["labels"][0][batch["labels"][0] != pad_token].numpy()
            ],
            **{
                "k"
                + str(i): [
                    idx2word[w] for w in topks[0][:, i][batch["labels"][0] != pad_token]
                ]
                for i in range(5)
            },
        }
    )
    df.loc[(df["kl"] > 0.1) | (df["true"] != df["k0"])]
    return df


