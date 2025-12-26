# src/datasets/build.py
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset

from .kaggle_dataset import KaggleImageDataset

from pathlib import Path
import importlib
from omegaconf import DictConfig, OmegaConf
import torch

import importlib
import inspect
from .transforms import TransformPipeline
from .base import import_target
from .main_dataset import MainDataset_Cached
from .preprocess import prepare_dataframe


import importlib

def resolve_dataset_class(name: str):
    """
    Resolve dataset class from:
    1) current namespace or dotted path
    2) src.datasets.<name>
    """

    # --------------------------------------------------
    # 1) Try as-is
    # --------------------------------------------------
    try:
        return import_target(name)
    except ImportError:
        pass

    # --------------------------------------------------
    # 2) Fallback to src.datasets.<name>
    # --------------------------------------------------
    try:
        return import_target(f"Colpo.dataset.{name}")
    except ImportError as e:
        raise ImportError(
            f"Could not resolve dataset '{name}'.\n"
            f"Tried:\n"
            f"  - {name}\n"
            f"  - Colpo.dataset.{name}"
        ) from e



def build_dataset(cfg, **args):
    
    ds_cfg = dict(cfg["dataset"])  # copy
    name = ds_cfg.pop("name")

    dataset_cls = resolve_dataset_class(name)

    # If dataset provides its own build logic, use it
    if hasattr(dataset_cls, "build") and callable(dataset_cls.build):
        return dataset_cls.build(ds_cfg, **args)

    # Fallback: direct constructor
    return dataset_cls(**ds_cfg, **args)



def save_indices(indices, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(indices, path)

def load_indices(path):
    return torch.load(path)

def build_train_dataset(dataset, cfg):
    idx_path = os.path.join(cfg.paths.indices_dir, "train_indices.pt")

    if os.path.exists(idx_path):
        indices = load_indices(idx_path)
    else:
        g = torch.Generator().manual_seed(cfg.training.seed)
        indices = torch.randperm(len(dataset), generator=g)
        save_indices(indices, idx_path)

    return Subset(dataset, indices)

def build_datasets(cfg):
    transform = build_transforms(cfg, "train")

    if cfg.dataset.name == "kaggle":
        train_ds = KaggleImageDataset(cfg.dataset.root, "Train", transform)
        val_ds   = KaggleImageDataset(cfg.dataset.root, "Validation", transform)
        test_ds  = KaggleImageDataset(cfg.dataset.root, "Test", transform)

    elif cfg.dataset.name == "colpo":
        df = pd.read_csv(cfg.dataset.df_path)

        train_df = df[df.split == "train"]
        val_df   = df[df.split == "val"]
        test_df  = df[df.split == "test"]

        train_ds = MainCachedDataset(
            train_df,
            cfg.dataset.image_root,
            cfg.dataset.image_col,
            cfg.dataset.label_col,
            cfg.dataset.extra_feature_cols,
            transform,
        )

        val_ds = MainCachedDataset(
            val_df,
            cfg.dataset.image_root,
            cfg.dataset.image_col,
            cfg.dataset.label_col,
            cfg.dataset.extra_feature_cols,
            transform,
        )

        test_ds = MainCachedDataset(
            test_df,
            cfg.dataset.image_root,
            cfg.dataset.image_col,
            cfg.dataset.label_col,
            cfg.dataset.extra_feature_cols,
            transform,
        )

    else:
        raise ValueError("Unknown dataset")

    return train_ds, val_ds, test_ds


def build_dataloaders(cfg):
    train_ds, val_ds, test_ds = build_datasets(cfg)

    # rule: train includes val
    full_train = ConcatDataset([train_ds, val_ds])

    train_loader = DataLoader(
        full_train,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,  # cached datasets
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader
