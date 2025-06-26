from torch.utils.data import Dataset, IterableDataset
import numpy as np
import torch


class TokenizerDataset(Dataset):
    def __init__(self, texts, tokenizer):
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return np.array(
            self.tokenizer.encode(self.texts[i], padding="max_length", max_length=384)
        )


# +
# class TokenizerPairIterableDataset(IterableDataset):
#     def __init__(self, dataset, tokenizer):
#         super().__init__()
#         self.dataset = dataset
#         self.tokenizer = tokenizer

#     def __iter__(self):
#         self.iter = iter(self.dataset)
#         return self

#     def __next__(self):
#         source_text, target_text, qid = next(self.iter)
      
#         # input_ids = np.array(
#         #     self.tokenizer.encode(pair[0], padding="max_length", max_length=256)
#         # )
#         # labels = np.array(
#         #     self.tokenizer.encode(
#         #         pair[1] + "[eos]", padding="max_length", max_length=448)
#         #     )
#         input_ids = np.array(
#             self.tokenizer.encode(source_text, padding="max_length", max_length=256)
#         )
#         labels = np.array(
#             self.tokenizer.encode(target_text + "[eos]", padding="max_length", max_length=448)
#         )
#         return {
#             "qid": qid,
#             "input_ids": input_ids,
#             "labels": labels,
#         }
# -

class TokenizerPairIterableDataset(IterableDataset):
    def __init__(self, dataset, tokenizer):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __iter__(self):
        self.iter = iter(self.dataset)
        return self
    def __next__(self):
            item = next(self.iter)
            # 如果 item 长度为 3，则正常解包；否则生成一个合成 qid
            if len(item) == 3:
                source_text, target_text, qid = item
            elif len(item) == 2:
                source_text, target_text = item
                qid = f"synthetic_{hash(source_text + target_text)}"
            else:
                raise ValueError("Item must be a tuple of length 2 or 3.")

            input_ids = np.array(
                self.tokenizer.encode(source_text, padding="max_length", max_length=256)
            )
            labels = np.array(
                self.tokenizer.encode(target_text + "[eos]", padding="max_length", max_length=448)
            )
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            return {
                "qid": qid,
                "input_ids": input_ids,
                "labels": labels,
            }


class TokenizerPairDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        super().__init__()
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    # def __getitem__(self, i):
    #     source_text, target_text, qid = self.pairs[i]  # 解包三元组

    #     input_ids = np.array(
    #         self.tokenizer.encode(
    #             self.pairs[i][0], padding="max_length", max_length=256
    #         )
    #     )
    #     labels = np.array(
    #         self.tokenizer.encode(
    #             self.pairs[i][1] + "[eos]", padding="max_length", max_length=448
    #         )
    #     )
    #     return {
    #         "qid": qid,
    #         "input_ids": input_ids,
    #         "labels": labels,
    #     }
    def __getitem__(self, i):
        # 解包三元组
        source_text, target_text, qid = self.pairs[i]
        input_ids = np.array(
            self.tokenizer.encode(source_text, padding="max_length", max_length=256)
        )
        labels = np.array(
            self.tokenizer.encode(target_text + "[eos]", padding="max_length", max_length=448)
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return {
            "qid": qid,
            "input_ids": input_ids,
            "labels": labels,
        }


class RandomlySampleDataset(IterableDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.i = 0
        self.indices = None

    def __iter__(self):
        self.indices = np.random.permutation(len(self.dataset))
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.dataset):
            self.i = 0
            self.indices = np.random.permutation(len(self.dataset))

        item = self.dataset[self.indices[self.i]]
        self.i = self.i + 1
        return item


class MixDataset(IterableDataset):
    def __init__(self, datasets, proportions):
        super().__init__()
        self.datasets = datasets
        self.dataset_iters = [None for x in self.datasets]
        self.proportions = np.cumsum(proportions)
        self.i = 0

    def __iter__(self):
        self.dataset_iters = [iter(d) for d in self.datasets]
        self.i = 0
        return self

    def __next__(self):
        item = None
        for i, p in enumerate(self.proportions):
            if self.i < p:
                try:
                    item = next(self.dataset_iters[i])
                except StopIteration:
                    self.dataset_iters[i] = iter(self.datasets[i])
                    item = next(self.dataset_iters[i])
                break

        self.i = (self.i + 1) % self.proportions[-1]
        assert item is not None
        return item


class Tokenizer_with_attention_mask(Dataset):
    def __init__(self, pairs, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.pairs = pairs
        self.padding_index = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        source_text, target_text, qid = self.pairs[i]
        sourece_encoding = self.tokenizer(
            source_text,
            max_length=256,
            padding='max_length', 
            return_tensors = 'pt',
            return_attention_mask = True,
            truncation=True
        )
        input_ids = sourece_encoding['input_ids'].squeeze(0)
        attention_mask = sourece_encoding['attention_mask'].squeeze(0)

        target_encoding = self.tokenizer(
            target_text,
            max_length = 448,
            padding = 'max_length',
            return_tensors = 'pt',
            truncation=True
        )
        labels = target_encoding['input_ids'].squeeze(0)
        # print(f"Item {i}, labels range: {labels.min().item()} to {labels.max().item()}")
        labels[labels == self.padding_index] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "qid": qid  # Include qid if you need it downstream
        }
