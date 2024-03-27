import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, embedding_size, head_size):
        super().__init__()
        self.kW = nn.Linear(embedding_size, head_size)
        self.vW = nn.Linear(embedding_size, head_size)
        self.qW = nn.Linear(embedding_size, head_size)

    def forward(self, k, v, q, mask):
        B, T, C = k.shape
        # h = head_size
        k = self.kW(k)  # (B,T,C) @ (C, h) -> (B, T, h)
        q = self.qW(q)  # (B,T,C) @ (C, h) -> (B, T, h)
        attn = q @ k.transpose(-1, -2) * C**-0.5  # (B,T,h) @ (B,h,T) -> (B,T,T)
        if mask is not None:
            # NOTE: masking should be done with mask[:Td, :Te]?
            attn = attn.masked_fill(mask[:T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        # NOTE: dropout goes here
        v = self.vW(v)  # (B,T,C) @ (C, h) -> (B, T, h)
        out = attn @ v  # (B,T,T) @ (B,T,h) -> (B,T,h)
        return out


class MultiheadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(embedding_size, head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embedding_size, embedding_size)
        # NOTE: dropout goes here

    def forward(self, k, v, q, mask):
        out = torch.cat([h(k=k, q=q, v=v, mask=mask) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.ReLU(),
            nn.Linear(embedding_size * 4, embedding_size),
            # NOTE: dropout goes here
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
        # NOTE: dropout goes here
        if self.is_decoder:
            self.cross_attn = MultiheadAttention(embedding_size, num_heads, head_size)
            self.ln2 = nn.LayerNorm(embedding_size)
            # NOTE: dropout goes here
        self.ffwd = FeedForward(embedding_size)
        self.ln3 = nn.LayerNorm(embedding_size)

    def forward(self, first_inp, first_mask, second_inp=None, second_mask=None):
        assert not (
            self.is_decoder and (second_inp is None or second_mask is None)
        ), "decoder needs second input and mask"
        _x = first_inp
        x = self.self_attn(k=first_inp, q=first_inp, v=first_inp, mask=first_mask)
        x = self.ln1(x + _x)
        if self.is_decoder:
            _x = x
            x = self.cross_attn(k=second_inp, v=second_inp, q=x, mask=second_mask)
            x = self.ln2(x + _x)
        _x = x
        x = self.ffwd(x)
        x = self.ln3(x + _x)
        return x


class Encoder(nn.Module):
    def __init__(self, embedding_size, layers, heads):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embedding_size, heads, is_decoder=False)
                for _ in range(layers)
            ]
        )

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, embedding_size, layers, heads):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(embedding_size, heads) for _ in range(layers)]
        )

    def forward(self, dec_out, dec_mask, enc_src, enc_mask):
        for layer in self.layers:
            x = layer(dec_out, dec_mask, enc_src, enc_mask)
        return x
