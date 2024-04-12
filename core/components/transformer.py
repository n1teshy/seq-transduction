import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import dropout


class Head(nn.Module):
    def __init__(self, embedding_size, head_size):
        super().__init__()
        self.kW = nn.Linear(embedding_size, head_size)
        self.vW = nn.Linear(embedding_size, head_size)
        self.qW = nn.Linear(embedding_size, head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k, v, q, mask=None):
        B, T, C = q.shape
        # h = head_size
        k = self.kW(k)  # (B, Te, C) @ (C, h) -> (B, Te, h)
        q = self.qW(q)  # (B, Td, C) @ (C, h) -> (B, Td, h)
        attn = (
            q @ k.transpose(-1, -2) * C**-0.5
        )  # (B, Td, h) @ (B, h, Te) -> (B, Td, Te)
        if mask is not None:
            attn = attn.masked_fill(mask[:, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        v = self.vW(v)  # (B, Te, C) @ (C, h) -> (B, Te, h)
        out = attn @ v  # (B, Td, Te) @ (B, Te, h) -> (B, Td, h)
        return out


class MultiheadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(embedding_size, head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k, v, q, mask=None):
        out = torch.cat([h(k=k, q=q, v=v, mask=mask) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.ReLU(),
            nn.Linear(embedding_size * 4, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, is_decoder=True):
        super().__init__()
        assert (
            embedding_size % num_heads == 0
        ), "embedding size must be divisible by number of heads"
        self.is_decoder = is_decoder
        head_size = embedding_size // num_heads
        self.self_attn = MultiheadAttention(embedding_size, num_heads, head_size)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.dropout1 = nn.Dropout(dropout)
        if self.is_decoder:
            self.cross_attn = MultiheadAttention(embedding_size, num_heads, head_size)
            self.ln2 = nn.LayerNorm(embedding_size)
            self.dropout2 = nn.Dropout(dropout)
        self.ffwd = FeedForward(embedding_size)
        self.ln3 = nn.LayerNorm(embedding_size)

    def forward(self, first_inp, first_mask=None, second_inp=None, second_mask=None):
        _x = first_inp
        x = self.self_attn(k=first_inp, q=first_inp, v=first_inp, mask=first_mask)
        x = self.dropout1(self.ln1(x + _x))
        if self.is_decoder:
            _x = x
            x = self.cross_attn(k=second_inp, v=second_inp, q=x, mask=second_mask)
            x = self.dropout2(self.ln2(x + _x))
        _x = x
        x = self.ffwd(x)
        x = self.ln3(x + _x)
        return x


class Encoder(nn.Module):
    def __init__(self, embedding_size, blocks, heads):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_size, heads, is_decoder=False)
                for _ in range(blocks)
            ]
        )

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(first_inp=x, first_mask=mask)
        return x


class Decoder(nn.Module):
    def __init__(self, embedding_size, blocks, heads):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(embedding_size, heads) for _ in range(blocks)]
        )

    def forward(self, dec_out, enc_in, dec_mask=None, enc_mask=None):
        for block in self.blocks:
            dec_out = block(
                first_inp=dec_out,
                first_mask=dec_mask,
                second_inp=enc_in,
                second_mask=enc_mask,
            )
        return dec_out
