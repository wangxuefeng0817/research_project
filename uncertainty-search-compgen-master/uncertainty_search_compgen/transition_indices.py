import itertools
from collections import defaultdict


def make_transition_indices(train_pairs, tokenizer):
    encodings = tokenizer.batch_encode_plus([x[1] + "[eos]" for x in train_pairs])
    accum = defaultdict(set)
    for x, y in itertools.chain.from_iterable(
        [
            list(zip([harness.model.config.decoder_start_token_id] + e[:-1], e))
            for e in encodings["input_ids"]
        ]
    ):
        accum[x].add(y)

    maxlen = np.max([len(v) for v in accum.values()])
    initial_lookup = torch.zeros(tokenizer.vocab_size, dtype=torch.long)
    initial_lookup[torch.tensor(sorted(list(accum.keys())))] = torch.arange(
        len(accum.keys())
    )
    initial_lookup = initial_lookup.to("cuda")
    permitted_transition_indices = torch.tensor(
        [
            list(accum[key]) + ([0] * (maxlen - len(accum[key])))
            for key in sorted(list(accum.keys()))
        ]
    )
    return permitted_transition_indices


import os
print(os.getcwd())


