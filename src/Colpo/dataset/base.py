# src/datasets/base.py
from typing import Tuple, Union
import torch
from torch.utils.data import Dataset
import importlib

Sample = Union[
    Tuple[torch.Tensor, int],
    Tuple[torch.Tensor, torch.Tensor, int]
]


def import_target(path: str):
    """
    Import class from dotted path.
    """
    module_path, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)

class BaseDataset(Dataset):
    def has_extra_features(self) -> bool:
        return False


class BasicDataset(Dataset):
    def __init__(self, features, labels, extras=None):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.extras = (
            torch.as_tensor(extras, dtype=torch.float32)
            if extras is not None else None
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.extras is not None:
            return self.features[idx], self.extras[idx], self.labels[idx]
        else:
            return self.features[idx], self.labels[idx]

