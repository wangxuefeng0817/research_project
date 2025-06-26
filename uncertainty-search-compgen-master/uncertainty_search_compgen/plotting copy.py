import matplotlib.pyplot as plt
import pandas as pd


def plot_batch_and_metric(tokenizer, batch, loss, idx2word, pad_token):
    fig, ax1 = plt.subplots(figsize=(18, 4))

    color = "tab:blue"
    ax1.set_ylabel("loss", color=color)  # we already handled the x-label with ax1
    ax1.set_xlabel("Ground truth")
    
    y = loss[0][batch["labels"][0] != pad_token]
    x = range(len(y))

    ax1.stem(x, y, label="loss", linefmt=color, markerfmt=color, basefmt="")
    ax1.tick_params(axis="y", labelcolor=color)

    idx2word = {i: w for w, i in tokenizer.vocab.items()}

    decoded_tokens = batch["labels"][0][batch["labels"][0] != pad_token].tolist()

    ax1.set_xticks(range(len(decoded_tokens)))
    ax1.set_xticklabels([idx2word[i] for i in decoded_tokens], rotation=45, ha="right")
    ax1.set_title(tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True))

    ax1.grid(axis='x', alpha=0.4)
    ax1.set_axisbelow(True)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def plot_batch_and_metric_comparison(tokenizer, batch, predicted_tokens, loss, idx2word, pad_token):
    
    # Decode the label
    decoded_tokens = batch["labels"][0][batch["labels"][0] != pad_token].tolist()
    decoded_tokens = [idx2word[i] for i in decoded_tokens]

    fig, ax = plt.subplots(figsize=(3, 12))
    x = loss[0][batch["labels"][0] != pad_token]
    y = range(len(x))
    
    # Plot
    ax.plot(x, y, 'D')
    ax.hlines(y, 0, x)
    for i in range(len(y)):
        ax.text(max(x)+1, i, decoded_tokens[i], va="center")

    # Labeling
    ax.set_yticks(range(len(decoded_tokens)))
    ax.set_yticklabels(decoded_tokens)

    # Final touch
    ax.grid(axis='x', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    plt.show()


def plot_batch_and_metric_pair(tokenizer, batch, kl, loss, idx2word, pad_token):
    fig, ax1 = plt.subplots(figsize=(18, 6))

    color = "tab:red"
    ax1.set_xlabel("token")
    ax1.set_ylabel("ent", color=color)
    
    y_kl = kl[0][batch["labels"][0] != pad_token]
    x = range(len(y_kl))
    ax1.stem(x, y_kl, label="kl", linefmt=color, markerfmt=color, basefmt="")

    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_title(tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True))

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel("loss", color=color)  # we already handled the x-label with ax1
    
    y_loss = loss[0][batch["labels"][0] != pad_token]
    ax2.stem(x, y_loss, label="loss", linefmt=color, markerfmt=color, basefmt="")

    ax2.tick_params(axis="y", labelcolor=color)

    idx2word = {i: w for w, i in tokenizer.vocab.items()}

    decoded_tokens = batch["labels"][0][batch["labels"][0] != pad_token].tolist()

    ax1.set_xticks(range(len(decoded_tokens)))
    ax1.set_xticklabels([idx2word[i] for i in decoded_tokens], rotation=45, ha="right")
    print("done")

    ax1.grid(axis='x', alpha=0.4)
    ax1.set_axisbelow(True)
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
