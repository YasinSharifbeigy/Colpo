# src/datasets/build.py
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset

from .kaggle_dataset import KaggleImageDataset

from pathlib import Path
import importlib
from omegaconf import DictConfig, OmegaConf
import torch

from . import TransformPipeline
from . import import_target

def build_dataset(cfg: DictConfig | dict):
    """
    Build dataset instance from config.
    """

    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    if "dataset" not in cfg:
        raise ValueError("Config must contain 'dataset' section")

    ds_cfg = cfg["dataset"]

    # --------------------------------------------------
    # Dataset class
    # --------------------------------------------------
    if "name" not in ds_cfg:
        raise ValueError("dataset.name must be specified")

    dataset_cls = import_target(
        f"src.datasets.{ds_cfg['name']}"
    )

    # --------------------------------------------------
    # Pipelines
    # --------------------------------------------------
    base_pipeline_cfg = ds_cfg.pop("base_pipeline", None)
    aug_pipeline_cfg = ds_cfg.pop("augmentation_pipeline", None)

    base_pipeline = (
        TransformPipeline(base_pipeline_cfg)
        if base_pipeline_cfg is not None
        else None
    )

    aug_pipeline = (
        TransformPipeline(aug_pipeline_cfg)
        if aug_pipeline_cfg is not None
        else None
    )

    # --------------------------------------------------
    # Remaining dataset args
    # --------------------------------------------------
    ds_kwargs = {
        k: v
        for k, v in ds_cfg.items()
        if k != "name"
    }

    # Inject pipelines
    ds_kwargs["base_pipeline"] = base_pipeline
    ds_kwargs["aug_pipeline"] = aug_pipeline

    # --------------------------------------------------
    # Instantiate dataset
    # --------------------------------------------------
    dataset = dataset_cls(**ds_kwargs)

    return dataset


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
