# src/datasets/transforms.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

import itertools
from omegaconf.dictconfig import DictConfig
from collections import defaultdict

# ============================================================
# 1) Transform Registry
# ============================================================

class TransformRegistry:
    """
    Registry with built-in base transforms.
    """
    def __init__(self, register_base: bool = True):
        self._registry = {}

        if register_base:
            self._register_base_transforms()

    def register(self, name, fn, override: bool = False):
        if name in self._registry and not override:
            raise KeyError(f"Transform '{name}' already registered")
        self._registry[name] = fn

    def get(self, name):
        if name not in self._registry:
            raise KeyError(
                f"Transform '{name}' not found. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def __contains__(self, name):
        return name in self._registry

    def keys(self):
        return self._registry.keys()

    def _register_base_transforms(self):
        """
        Base transforms are ALWAYS available.
        """

        from .basic_transforms import (
            to_tensor,
            normalize,
            resize,
            crop_to_square,
            crop,
            random_crop,
            shift,
            random_shift,
            rotation,
            random_rotation,
            random_h_flip,
            random_v_flip,
            set_brightness,
            random_brightness,
            color_jitter,
        )

        # BASIC
        self._registry["to_tensor"] = to_tensor
        self._registry["normalize"] = normalize

        # GEOMETRIC
        self._registry["resize"] = resize
        self._registry["crop_to_square"] = crop_to_square
        self._registry["crop"] = crop
        self._registry["random_crop"] = random_crop
        self._registry["shift"] = shift
        self._registry["random_shift"] = random_shift
        self._registry["rotation"] = rotation
        self._registry["random_rotation"] = random_rotation

        # FLIPS
        self._registry["random_h_flip"] = random_h_flip
        self._registry["random_v_flip"] = random_v_flip

        # COLOR
        self._registry["set_brightness"] = set_brightness
        self._registry["random_brightness"] = random_brightness

        # FULL PIPELINE
        self._registry["color_jitter"] = color_jitter




class AugmentationPipeline():
    def __init__(self, cfg: DictConfig|dict|None, transform_registry: TransformRegistry|None = None ):
        self.cfg = cfg
        self.transform_registry = transform_registry if transform_registry is not None else TransformRegistry()
        self.num_random_augmentations = 0
        self.num_ecplicit_augmentations = 0
        self.explicit_grid = self.build_explicit_grid()
        self.pipeline = self.build_pipeline()
        
    
    def build_explicit_grid(self):
        """
        Builds grid ONLY over explicit arguments.

        Output example:
        [
        {'rotation': {'angle': 0}, 'color_jitter': {'contrast': 0.9}},
        ...
        ]
        """

        # (transform_name, arg_name) -> args
        explicit_args = dict()
        self.num_ecplicit_augmentations = 0
        self.num_random_augmentations = 0

        for transform_name, transform_cfg in self.cfg.items():
            for arg_name, spec in transform_cfg.items():
                if spec["type"] == "explicit":
                    explicit_args[(transform_name, arg_name)] = spec["args"]
                    self.num_ecplicit_augmentations += 1
                elif spec["type"] == "random":
                    self.num_random_augmentations += 1
                    

        if not explicit_args:
            return [dict()]  # important: length=1 grid

        keys = list(explicit_args.keys())
        values = [explicit_args[k] for k in keys]

        grid = []
        for combo in itertools.product(*values):
            sample = defaultdict(dict)
            for (transform_name, arg_name), value in zip(keys, combo):
                sample[transform_name][arg_name] = value
            grid.append(dict(sample))

        return grid
    
    def resolve_args(self, pipline_step, idx):
        """
        For a given idx:
        - explicit args → choose value from grid
        - random args → pass args AS-IS to transform
        """
        resolved = {}
        
        if idx is None:
            grid_idx = 0
        else:
            grid_idx = idx % len(self.explicit_grid)

        transform_cfg = pipline_step['cfg']
        transform_name = pipline_step['name']

        for arg_name, spec in transform_cfg.items():
            arg_type = spec["type"]

            if arg_type == "explicit":
                resolved[arg_name] = self.explicit_grid[grid_idx][transform_name][arg_name]

            elif arg_type == "random" or arg_type == "simple":
                resolved[arg_name] = spec["args"]

            else:
                raise ValueError(
                    f"Unknown arg type '{arg_type}' for argument '{arg_name}'"
                )

        return resolved

    def build_pipeline(self):
        """
        Builds ordered augmentation pipeline.

        cfg.augmentations:
            transform_name -> transform_cfg
        """
        pipeline = []

        for transform_name, transform_cfg in self.cfg.items():
            # print(transform_name)
            if transform_name not in self.transform_registry:
                raise KeyError(
                    f"Transform '{transform_name}' not registered. "
                    f"Available: {list(self.transform_registry._registry.keys())}"
                )

            transform_fn = self.transform_registry.get(transform_name)

            pipeline.append({
                "name": transform_name,
                "fn": transform_fn,
                "cfg": transform_cfg,
            })

        return pipeline

    def __call__(self, img, idx=None):
        for step in self.pipeline:
            kwargs = self.resolve_args(step, idx)
            img = step["fn"](img, **kwargs)

        return img


# ============================================================
# 2) Explicit grid builder
# ============================================================

def build_explicit_grid(transform_cfg):
    """
    Builds grid ONLY over explicit arguments.

    transform_cfg:
        arg_name -> {type, args}
    """
    explicit_args = {
        name: spec["args"]
        for name, spec in transform_cfg.items()
        if spec["type"] == "explicit"
    }

    if not explicit_args:
        return [dict()]  # important: length=1 grid

    keys = list(explicit_args.keys())
    values = [explicit_args[k] for k in keys]

    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*values)
    ]


# ============================================================
# 3) Argument resolver (YOUR exact semantics)
# ============================================================

def resolve_args(transform_cfg, explicit_grid, idx):
    """
    For a given idx:
      - explicit args → choose value from grid
      - random args → pass args AS-IS to transform
    """
    resolved = {}

    # idx may be None (e.g. validation)
    if idx is None:
        grid_idx = 0
    else:
        grid_idx = idx % len(explicit_grid)

    explicit_vals = explicit_grid[grid_idx]

    for arg_name, spec in transform_cfg.items():
        arg_type = spec["type"]

        if arg_type == "explicit":
            resolved[arg_name] = explicit_vals[arg_name]

        elif arg_type == "random":
            resolved[arg_name] = spec["args"]

        else:
            raise ValueError(
                f"Unknown arg type '{arg_type}' for argument '{arg_name}'"
            )

    return resolved


# ============================================================
# 4) Pipeline builder
# ============================================================

def build_pipeline(cfg, transform_registry):
    """
    Builds ordered augmentation pipeline.

    cfg.augmentations:
        transform_name -> transform_cfg
    """
    pipeline = []

    if not hasattr(cfg, "augmentations"):
        return pipeline

    for transform_name, transform_cfg in cfg.augmentations.items():
        if transform_name not in transform_registry:
            raise KeyError(
                f"Transform '{transform_name}' not registered. "
                f"Available: {list(transform_registry._registry.keys())}"
            )

        transform_fn = transform_registry.get(transform_name)
        explicit_grid = build_explicit_grid(transform_cfg)

        pipeline.append({
            "name": transform_name,
            "fn": transform_fn,
            "cfg": transform_cfg,
            "grid": explicit_grid,
        })

    return pipeline


# ============================================================
# 5) Apply pipeline
# ============================================================

def apply_pipeline(img, pipeline, idx=None):
    """
    Applies transforms in order.
    """
    for step in pipeline:
        kwargs = resolve_args(step["cfg"], step["grid"], idx)
        img = step["fn"](img, **kwargs)
    return img
