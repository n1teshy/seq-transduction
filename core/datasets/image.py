import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from core.config import device, TOKEN_PAD, TOKEN_BOS, TOKEN_EOS


class OCRDataset(Dataset):
    def __init__(self, folder, mapping_file, tokenizer):
        super().__init__()
        self.folder = folder
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.special_tokens[TOKEN_PAD]
        self.bos_id = tokenizer.special_tokens[TOKEN_BOS]
        self.eos_id = tokenizer.special_tokens[TOKEN_EOS]
        self.img_to_tokens = self.get_mapping(mapping_file)
        self.filenames = self.get_file_names()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        tokens = self.img_to_tokens[filename]
        image = Image.open(os.path.join(self.folder, filename)).convert("L")
        return (image, tokens)

    def get_mapping(self, mapping_file):
        mapping = {}
        for line in open(os.path.join(self.folder, mapping_file)):
            line = line.strip("\n")
            splits = line.split(":", maxsplit=1)
            mapping[splits[0]] = (
                [self.bos_id] + self.tokenizer.encode(splits[1]) + [self.eos_id]
            )
        return mapping

    def get_file_names(self):
        _, __, files = next(os.walk(self.folder))
        return files

    def collate(self, samples):
        images, tokens = [s[0] for s in samples], [s[1] for s in samples]
        max_height = max(image.height for image in images)
        max_width = max(image.width for image in images)
        transform = transforms.Compose(
            [
                transforms.Resize((max_height, max_width)),
                transforms.ToTensor(),
            ]
        )
        tokens = [torch.tensor(t) for t in tokens]
        tokens = pad_sequence(tokens, padding_value=self.pad_id, batch_first=True)
        images = torch.stack([transform(image) for image in images], dim=0)
        return images.to(device), tokens.to(device)
