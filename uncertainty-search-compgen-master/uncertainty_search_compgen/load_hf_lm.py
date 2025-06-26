from transformers import AutoTokenizer


def load_hf_tokenizer(model_name_or_path: str = None):
    model_name_or_path = model_name_or_path or "Salesforce/codet5p-220m"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=False, cache_dir="/tmp/hf"
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    idx2word = {i: w for w, i in tokenizer.vocab.items()}

    return tokenizer, idx2word




