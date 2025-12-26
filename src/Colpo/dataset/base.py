# src/datasets/base.py
from typing import Tuple, Union
import torch
from torch.utils.data import Dataset
import importlib
import inspect


Sample = Union[
    Tuple[torch.Tensor, int],
    Tuple[torch.Tensor, torch.Tensor, int]
]


def import_target(name: str):
    """
    Import class from dotted path or current globals.
    """
    # --------------------------------------------------
    # 1) Already in globals (Jupyter / same file)
    # --------------------------------------------------
    if name in globals():
        obj = globals()[name]
        if inspect.isclass(obj):
            return obj

    # --------------------------------------------------
    # 2) Explicit dotted path
    # --------------------------------------------------
    if "." in name:
        try:
            module_path, cls_name = name.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, cls_name)
        except Exception as e:
            raise ImportError(
                f"Failed to import '{name}' as dotted path"
            ) from e

    # --------------------------------------------------
    # 3) Nothing worked → FAIL LOUDLY
    # --------------------------------------------------
    raise ImportError(
        f"import_target could not resolve '{name}'. "
        f"Expected a dotted path or a global class."
    )


class BaseDataset(Dataset):

    @classmethod
    def build(cls, cfg: dict, **extra):
        """
        Build dataset from config.

        Default behavior: pass cfg as kwargs.
        Subclasses may override.
        """
        return cls(**cfg, **extra)
    
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

