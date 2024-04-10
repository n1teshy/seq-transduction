import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from core.config import device


class ImageDataset(Dataset):
    def __init__(self, folder, mapping_file="mapping.txt"):
        super().__init__()
        self.folder = folder
        self.mapping = self.get_mapping(mapping_file)
        self.filenames = self.get_file_names()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        label = self.mapping[filename]
        image = self.transform(
            Image.open(os.path.join(self.folder, filename)).convert("L")
        )
        return image.to(device), torch.tensor(label)

    def get_mapping(self, mapping_file):
        mapping = {}
        for line in open(os.path.join(self.folder, mapping_file)):
            line = line.strip("\n")
            splits = line.split(":", maxsplit=1)
            if len(splits) != 2:
                raise ValueError(f"found more than two splits for '{line}'")
            mapping[splits[0]] = int(splits[1])
        return mapping

    def get_file_names(self):
        _, __, files = next(os.walk(self.folder))
        return files
