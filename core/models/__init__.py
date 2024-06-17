import torch
import torch.nn as nn

from core.config import device
from core.components.cnn import BasicBlock, BottleNeck, ResNet
from core.components.embedding import TokenEmbedding
from core.components.transformer import Encoder, Decoder


def resnet14(in_channels=1, num_classes=100):
    return ResNet(
        BasicBlock, [1, 1, 1, 1], in_chanels=in_channels, num_classes=num_classes
    )


def resnet17(in_channels=1, num_classes=100):
    return ResNet(
        BasicBlock, [2, 2, 2, 1], in_chanels=in_channels, num_classes=num_classes
    )


def resnet18(in_channels=1, num_classes=100):
    return ResNet(
        BasicBlock, [2, 2, 2, 2], in_chanels=in_channels, num_classes=num_classes
    )


def resnet34(in_channels=1, num_classes=100):
    return ResNet(
        BasicBlock, [3, 4, 6, 3], in_chanels=in_channels, num_classes=num_classes
    )


def resnet50(in_channels=1, num_classes=100):
    return ResNet(
        BottleNeck, [3, 4, 6, 3], in_chanels=in_channels, num_classes=num_classes
    )


def resnet101(in_channels=1, num_classes=100):
    return ResNet(
        BottleNeck, [3, 4, 23, 3], in_chanels=in_channels, num_classes=num_classes
    )


def resnet152(in_channels=1, num_classes=100):
    return ResNet(
        BottleNeck, [3, 8, 36, 3], in_chanels=in_channels, num_classes=num_classes
    )


class Transformer(nn.Module):
    def __init__(
        self,
        in_vocab_size,
        out_vocab_size,
        embedding_size,
        max_len,
        enc_layers,
        dec_layers,
        enc_heads,
        dec_heads,
        src_pad_id,
        tgt_pad_id,
    ):
        super().__init__()
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.in_emb = TokenEmbedding(in_vocab_size, embedding_size, max_len)
        self.out_emb = TokenEmbedding(out_vocab_size, embedding_size, max_len)
        self.encoder = Encoder(embedding_size, enc_layers, enc_heads)
        self.decoder = Decoder(embedding_size, dec_layers, dec_heads)
        self.linear = nn.Linear(embedding_size, out_vocab_size)

    def forward(self, x, y):
        x_mask, y_mask = self.get_masks(x, y)
        x_emb, y_emb = self.in_emb(x), self.out_emb(y)
        x_enc = self.encoder(x_emb, x_mask)
        out = self.decoder(
            dec_out=y_emb, dec_mask=y_mask, enc_in=x_enc, enc_mask=x_mask
        )
        return self.linear(out)

    def get_masks(self, x, y):
        x_mask = (x != self.src_pad_id).unsqueeze(1)
        tgt_seq_len = y.shape[1]
        y_pad_mask = (y != self.tgt_pad_id).unsqueeze(1)
        y_lh_mask = torch.tril(
            torch.ones(tgt_seq_len, tgt_seq_len, dtype=torch.long, device=device)
        )
        return x_mask, (y_pad_mask & y_lh_mask)

    @staticmethod
    def spawn(*args, **kwargs):
        cache = kwargs.get("cache", None)
        if cache is not None:
            del kwargs["cache"]
        model = Transformer(*args, **kwargs)
        model = model.to(device)
        if cache is not None:
            model.load_state_dict(torch.load(cache, map_location=device))
        return model


class OCR(nn.Module):
    def __init__(
        self,
        encoder,
        out_vocab_size,
        embedding_size,
        max_len,
        dec_layers,
        dec_heads,
        tgt_pad_id,
    ):
        super().__init__()
        self.encoder = encoder
        self.tgt_pad_id = tgt_pad_id
        self.embedding = TokenEmbedding(out_vocab_size, embedding_size, max_len)
        self.decoder = Decoder(embedding_size, dec_layers, dec_heads)
        self.linear = nn.Linear(embedding_size, out_vocab_size)

    def forward(self, pixels, tokens):
        mask = self.get_mask(tokens)
        emb_tokens = self.embedding(tokens)
        x_enc = self.encoder(pixels).unsqueeze(1)
        out = self.decoder(dec_out=emb_tokens, dec_mask=mask, enc_in=x_enc)
        return self.linear(out)

    def get_mask(self, text):
        tgt_seq_len = text.shape[1]
        y_pad_mask = (text != self.tgt_pad_id).unsqueeze(1)
        y_lh_mask = torch.tril(
            torch.ones(tgt_seq_len, tgt_seq_len, dtype=torch.long, device=device)
        )
        return y_pad_mask & y_lh_mask

    @staticmethod
    def spawn(*args, **kwargs):
        model = OCR(*args, **kwargs)
        return model.to(device)
