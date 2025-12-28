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
from torch.utils.data import Subset

import os
import pandas as pd

from .base import import_target




def resolve_preprocessor_name(name: str):
    """
    Resolve a preprocessor class.

    Resolution order:
    1) Explicit dotted path or global symbol via import_target
    2) Colpo.dataset.preprocess.<name>
    """

    # --------------------------------------------------
    # 1) Explicit dotted path or already-imported symbol
    # --------------------------------------------------
    try:
        return import_target(name)
    except Exception as e1:
        pass

    # --------------------------------------------------
    # 2) Project default preprocessors
    # --------------------------------------------------
    try:
        module = importlib.import_module("Colpo.dataset.preprocess")
        return getattr(module, name)
    except Exception as e2:
        raise ImportError(
            f"Could not resolve preprocessor '{name}'.\n"
            f"Tried:\n"
            f"  - explicit import_target('{name}')\n"
            f"  - Colpo.dataset.preprocess.{name}"
        ) from e2


def build_preprocessor(cfg, *kargs):
    """
    Build (and optionally fit/load) a Preprocessor instance.

    Behavior is fully controlled by cfg.preprocess.
    """
    pp_cfg = cfg.preprocess
    pp_cfg = dict(pp_cfg)  # defensive copy

    # Resolve class
    name = pp_cfg.pop("name")
    PreprocessCls = resolve_preprocessor_name(name)

    # Persistence config
    save_path = pp_cfg.pop("save_path", None)
    load_if_exists = pp_cfg.pop("load_if_exists", False)
    load_path = pp_cfg.pop("load_path", None)
    do_fit = pp_cfg.pop("do_fit", False)

    # If load_path is None, fall back to save_path
    if load_path is None:
        load_path = save_path

    # Fit arguments (optional)
    fit_args = pp_cfg.pop("fit_args", None)

    # Remaining keys are constructor kwargs
    init_kwargs = pp_cfg

    # Load existing preprocess if allowed
    if load_if_exists and load_path is not None and os.path.exists(load_path):
        preprocessor = PreprocessCls.load(load_path)

        # If we are not allowed to refit, return immediately
        if not do_fit:
            return preprocessor

    else:
        preprocessor = PreprocessCls(**init_kwargs)

    # Fit (explicit, controlled)
    if do_fit:
        if fit_args is None:
            raise ValueError(
                "do_fit=True but no fit_args provided in preprocess config"
            )

        # NOTE: we intentionally do IO here, not in __init__
        main_df = pd.read_excel(
            fit_args["main_dataframe"],
            keep_default_na=False,
            na_values=[""],
            dtype=str,
        )
        hpv_df = pd.read_excel(
            fit_args["hpv_dataframe"]
        )

        preprocessor.fit(main_df, hpv_df)

        # Save after successful fit
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            preprocessor.save(save_path)

    return preprocessor



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
        


def build_dataset(cfg, preprocessor= None, **args):
    
    ds_cfg = dict(cfg["dataset"])  # copy
    name = ds_cfg.pop("name")

    dataset_cls = resolve_dataset_class(name)
    
    for k in list(args.keys()):
        if k in ds_cfg:
            ds_cfg[k] = args.pop(k)
    
    if hasattr(dataset_cls, "requires_preprocess") and callable(dataset_cls.requires_preprocess):
        if dataset_cls.requires_preprocess():
            preprocess_kwargs = dataset_cls.preprocess(
                ds_cfg,
                preprocessor,
                **args
            )
            args.update(preprocess_kwargs)

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
    transform = None
    if cfg.dataset.name == "kaggle":
        train_ds = KaggleImageDataset(cfg.dataset.root, "Train", transform)
        val_ds   = KaggleImageDataset(cfg.dataset.root, "Validation", transform)
        test_ds  = KaggleImageDataset(cfg.dataset.root, "Test", transform)

    elif cfg.dataset.name == "colpo":
        pass

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
