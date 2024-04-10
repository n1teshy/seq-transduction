import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from core.config import TOKEN_BOS, TOKEN_EOS, TOKEN_PAD, device


class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_tokenizer, tgt_tokenizer):
        super().__init__()
        self.source = [
            src_tokenizer.encode(line)
            for line in open(src_file, encoding="utf-8").read().splitlines()
        ]
        self.target = [
            tgt_tokenizer.encode(line)
            for line in open(tgt_file, encoding="utf-8").read().splitlines()
        ]
        if len(self.source) != len(self.target):
            raise ValueError(
                f"Source and target datasets have different lengths, {len(self.source)}/{len(self.target)}"
            )
        self.src_bos_id, self.tgt_bos_id = (
            src_tokenizer.special_tokens[TOKEN_BOS],
            tgt_tokenizer.special_tokens[TOKEN_BOS],
        )
        self.src_eos_id, self.tgt_eos_id = (
            src_tokenizer.special_tokens[TOKEN_EOS],
            tgt_tokenizer.special_tokens[TOKEN_EOS],
        )
        self.src_pad_id, self.tgt_pad_id = (
            src_tokenizer.special_tokens[TOKEN_PAD],
            tgt_tokenizer.special_tokens[TOKEN_PAD],
        )

    def collate(self, batch):
        src_batch, tgt_batch = [], []
        for src_item, tgt_item in batch:
            src_batch.append(
                torch.tensor(
                    [self.src_bos_id, *src_item, self.src_eos_id], device=device
                )
            )
            tgt_batch.append(
                torch.tensor(
                    [self.tgt_bos_id, *tgt_item, self.tgt_eos_id], device=device
                )
            )
        src_batch = pad_sequence(
            src_batch, padding_value=self.src_pad_id, batch_first=True
        )
        tgt_batch = pad_sequence(
            tgt_batch, padding_value=self.tgt_pad_id, batch_first=True
        )
        return src_batch, tgt_batch

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return (self.source[index], self.target[index])
