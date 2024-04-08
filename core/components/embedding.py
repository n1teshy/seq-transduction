import torch
import torch.nn as nn

from core.config import device


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embedding_size)
        # NOTE: use device param to get the pos_emb tensor on device
        self.pos_emb = torch.zeros(max_len, embedding_size, device=device)
        self.pos_emb.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, embedding_size, step=2, device=device).float()

        self.pos_emb[:, 0::2] = torch.sin(pos / (10000 ** (_2i / embedding_size)))
        self.pos_emb[:, 1::2] = torch.cos(pos / (10000 ** (_2i / embedding_size)))

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb[:T, :]
        return tok_emb + pos_emb
