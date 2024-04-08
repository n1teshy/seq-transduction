import torch
import torch.nn as nn

from core.config import device
from core.components.cnn import ConvLayer
from core.components.embedding import TokenEmbedding
from core.components.transformer import Encoder, Decoder


class ConvNet(nn.Module):
    def __init__(self, final_channels, ep=1e-7):
        super().__init__()
        self.epsilon = ep
        self.conv1 = ConvLayer(1, 16, 4)
        self.conv2 = ConvLayer(16, 32, 3)
        self.conv3 = ConvLayer(32, 64, 2)
        self.conv4 = ConvLayer(64, 128, 2, stride=2)
        self.conv5 = ConvLayer(128, 256, 2, pool=2)
        self.conv6 = ConvLayer(256, final_channels, 2, stride=2)

    def forward(self, x):
        # normalize pixel values
        mean = torch.mean(x, dim=(-1, -2), keepdim=True)
        std = torch.std(x, dim=(-1, -2), keepdim=True)
        x = (x - mean) / (std + self.epsilon)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # mean operation will output (BATCH_SIZE, EMBEDDING_SIZE), does it make sense?
        return x.mean(dim=(-1, -2))


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
        out = self.decoder(y_emb, y_mask, x_enc, x_mask)
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
        cache = None
        if "cache" in kwargs:
            cache = kwargs["cache"]
            del kwargs["cache"]
        model = Transformer(*args, **kwargs)
        model = model.to(device)
        if cache is not None:
            model.load_state_dict(torch.load(cache, map_location=device))
        return model
