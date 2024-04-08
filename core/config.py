import torch


TOKEN_BOS, TOKEN_EOS, TOKEN_PAD, TOKEN_UNK = "<bos>", "<eos>", "<pad>", "<unk>"
special_tokens = [TOKEN_BOS, TOKEN_EOS, TOKEN_PAD, TOKEN_UNK]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout = 0.2
