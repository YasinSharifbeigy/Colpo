# src/datasets/colpo_cached_dataset.py
import torch
from PIL import Image
from .base import BaseDataset
from pandas.core.frame import DataFrame
import os

from .transforms import TransformPipeline


class MainDataset_Cached(BaseDataset):
    
    @classmethod
    def build(cls, cfg: dict, **extra_kwargs):
        """
        Build MainDataset_Cached from config.
        """

        # Extract dataset-specific pipeline configs
        base_pipeline_cfg = cfg.pop("base_pipeline", None)
        aug_pipeline_cfg  = cfg.pop("augmentation_pipeline", None)

        # Build pipelines
        base_pipeline = (
            TransformPipeline(base_pipeline_cfg)
            if base_pipeline_cfg is not None else None
        )

        augmentation_pipeline = (
            TransformPipeline(aug_pipeline_cfg)
            if aug_pipeline_cfg is not None else None
        )

        # Forward everything to constructor
        return cls(
            base_pipeline=base_pipeline,
            augmentation_pipeline=augmentation_pipeline,
            **cfg,
            **extra_kwargs,
        )

    def __init__(
        self,
        Data_info: DataFrame|None = None, 
        image_root: str|None = r'D:\Data\Cropped Folder',
        image_col = 'jpg_file', 
        label_col = 'Abnormality(Impression)',
        extra_feature_cols=None,
        base_images = None,
        labels = None,
        extras=None,

        aug_mode="none",                 # none | fully_cached

        base_pipeline=None,              # always applied
        augmentation_pipeline: TransformPipeline|None = None,               # optional augmentation pipeline

        num_random_augs=1,               # only for random modes
        include_original=True,            # include base-only sample
    ):
        assert aug_mode in {
            "none", "fully_cached"
        }

        assert labels is not None or Data_info is not None
        assert base_images is not None or image_root is not None
        assert extras is not None or not ((extra_feature_cols is not None) and (Data_info is None))

        self.base_images = [] if base_images is None else base_images
        self.labels = [] if labels is None else labels
        self.use_extra = extras is not None or extra_feature_cols is not None
        self.extras = [] if extras is None else extras
        
        if base_images is None:
            for _, row in Data_info.iterrows():
                img = Image.open(os.path.join(image_root,row['Patient ID'], row[image_col])).convert("RGB")
                self.base_images.append(img)

        if labels is None:
            for _, row in Data_info.iterrows():
                self.labels.append(torch.tensor(row[label_col] == 'True', dtype=torch.long))

                if self.use_extra and extras is None:
                    feats = torch.tensor(
                        row[extra_feature_cols].values,
                        dtype=torch.float32
                    )
                    self.extras.append(feats)

        self.aug_mode = aug_mode
        self.base_pipeline = base_pipeline
        self.augmentation_pipeline = augmentation_pipeline
        self.num_random_augs = num_random_augs
        self.include_original = include_original

        self.samples = []          # used for cached modes
        self._build_dataset()

    # --------------------------------------------------
    # Dataset construction
    # --------------------------------------------------

    def has_extra_features(self) -> bool:
        return self.use_extra
    
    def _apply_base(self, img):
        if self.base_pipeline is not None:
            return self.base_pipeline(img, idx=0)
        return img

    def _apply_aug(self, img, idx):
        if self.augmentation_pipeline is not None:
            return self.augmentation_pipeline(img, idx)
        return img

    def _build_dataset(self):
        n_images = len(self.base_images)

        # ----------------------------------------------
        # NONE
        # ----------------------------------------------
        if self.aug_mode == "none":
            for i, img in enumerate(self.base_images):
                img = self._apply_base(img.copy())
                self.samples.append(self._pack(img, i))

            self.num_base_images = n_images
            self.num_samples = len(self.samples)

            print(f"[MainCachedDataset] Total samples: {len(self.samples)}")
            return

        # ----------------------------------------------
        # EXPLICIT CACHE
        # ----------------------------------------------
        if self.aug_mode == "fully_cached":
            if self.augmentation_pipeline is None:
                raise ValueError("fully_cached requires aug_pipeline")

            # compute explicit grid size (product over transforms)
            total_explicit = len(self.augmentation_pipeline.explicit_grid)

            for i, img in enumerate(self.base_images):
                base_img = self._apply_base(img.copy())

                # original (no augmentation)
                if self.include_original:
                    self.samples.append(self._pack(base_img, i))

                # explicit augmented versions
                for k in range(total_explicit):
                    explicit_idx = i * total_explicit + k
                    if self.augmentation_pipeline.num_random_augmentations:
                        for _ in range(self.num_random_augs):
                            aug_img = self._apply_aug(base_img, explicit_idx)
                            self.samples.append(self._pack(aug_img, i))
            
            self.num_base_images = n_images
            self.num_samples = len(self.samples)

            print(
                f"[MainCachedDataset] Mode: fully_cached | "
                f"Images: {n_images} | "
                f"Explicit variants/image: {total_explicit} | "
                f"Include original: {self.include_original} | "
                f"Total samples: {len(self.samples)}"
            )
            return


    def _pack(self, img, i):
        if self.use_extra:
            return img, self.extras[i], self.labels[i]
        return img, self.labels[i]

    # --------------------------------------------------
    # PyTorch API
    # --------------------------------------------------

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # ----------------------------------------------
        # CACHED MODES
        # ----------------------------------------------
        return self.samples[idx]
