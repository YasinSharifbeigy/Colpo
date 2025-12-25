# src/datasets/colpo_cached_dataset.py
import torch
from PIL import Image
from .base import BaseDataset
from pandas.core.frame import DataFrame
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class MainCachedDataset(BaseDataset):
    def __init__(
        self,
        df,
        image_root,
        image_col,
        label_col,
        extra_feature_cols=None,
        transform=None,
    ):
        self.transform = transform
        self.use_extra = extra_feature_cols is not None

        self.images = []
        self.labels = []
        self.extra_features = [] if self.use_extra else None

        for _, row in df.iterrows():
            img = Image.open(
                f"{image_root}/{row[image_col]}"
            ).convert("RGB")

            if transform:
                img = transform(img)

            self.images.append(img)
            self.labels.append(int(row[label_col]))

            if self.use_extra:
                feats = torch.tensor(
                    row[extra_feature_cols].values,
                    dtype=torch.float32
                )
                self.extra_features.append(feats)

    def has_extra_features(self):
        return self.use_extra

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_extra:
            return self.images[idx], self.extra_features[idx], self.labels[idx]
        else:
            return self.images[idx], self.labels[idx]

# src/datasets/main_dataset.py
import torch

class MainCachedDataset:
    def __init__(
        self,
        base_images,
        image_root, 
        label_col = 'Abnormality(Impression)',
        extra_feature_cols=None,
        aug_mode="none",
        explicit_transform=None,
        aug_grid=None,
    ):  
        
        self.use_extra = extra_feature_cols is not None
        self.labels = []
        self.extras = []
        self.aug_mode = aug_mode

        self.samples = []

        if aug_mode == "explicit_cache":
            for i, img in enumerate(base_images):
                for params in aug_grid:
                    aug_img = explicit_transform(img, params)
                    self.samples.append((aug_img, extras[i] if extras else None, labels[i]))

        elif aug_mode == "index_dependent":
            self.base_images = base_images
            self.aug_grid = aug_grid
            self.explicit_transform = explicit_transform

        else:  # none or random_online
            self.samples = [(img, extras[i] if extras else None, labels[i])
                            for i, img in enumerate(base_images)]

    def __len__(self):
        if self.aug_mode == "index_dependent":
            return len(self.base_images) * len(self.aug_grid)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.aug_mode == "index_dependent":
            img_idx = idx // len(self.aug_grid)
            aug_idx = idx % len(self.aug_grid)
            img = self.base_images[img_idx]
            img = self.explicit_transform(img, self.aug_grid[aug_idx])
            extra = self.extras[img_idx] if self.extras else None
            return (img, extra, self.labels[img_idx]) if extra is not None else (img, self.labels[img_idx])

        img, extra, label = self.samples[idx]
        return (img, extra, label) if extra is not None else (img, label)

from torch.utils.data import Dataset
from .transforms import AugmentationPipeline


class MainCachedDataset(Dataset):
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
        aug_pipeline: AugmentationPipeline|None = None,               # optional augmentation pipeline

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
        self.aug_pipeline = aug_pipeline
        self.num_random_augs = num_random_augs
        self.include_original = include_original

        self.samples = []          # used for cached modes
        self._build_dataset()

    # --------------------------------------------------
    # Dataset construction
    # --------------------------------------------------

    def _apply_base(self, img):
        if self.base_pipeline is not None:
            return self.base_pipeline(img, idx=0)
        return img

    def _apply_aug(self, img, idx):
        if self.aug_pipeline is not None:
            return self.aug_pipeline(img, idx)
        return img

    def _build_dataset(self):
        n_images = len(self.base_images)

        # ----------------------------------------------
        # NONE
        # ----------------------------------------------
        if self.aug_mode == "none":
            for i, img in enumerate(self.base_images):
                img = self._apply_base(img)
                self.samples.append(self._pack(img, i))

            print(f"[MainCachedDataset] Total samples: {len(self.samples)}")
            return

        # ----------------------------------------------
        # EXPLICIT CACHE
        # ----------------------------------------------
        if self.aug_mode == "fully_cached":
            if self.aug_pipeline is None:
                raise ValueError("fully_cached requires aug_pipeline")

            # compute explicit grid size (product over transforms)
            total_explicit = len(self.aug_pipeline.explicit_grid)

            for i, img in enumerate(self.base_images):
                base_img = self._apply_base(img)

                # original (no augmentation)
                if self.include_original:
                    self.samples.append(self._pack(base_img, i))

                # explicit augmented versions
                for k in range(total_explicit):
                    explicit_idx = i * total_explicit + k
                    if self.aug_pipeline.num_random_augmentations:
                        for _ in range(self.num_random_augs):
                            aug_img = self._apply_aug(base_img, explicit_idx)
                            self.samples.append(self._pack(aug_img, i))

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
        return len(self.samples)

    def __getitem__(self, idx):
        # ----------------------------------------------
        # CACHED MODES
        # ----------------------------------------------
        return self.samples[idx]




class Imputation:
    def __init__(self):
        # Learned statistics
        self.age_median = None
        self.age_scaler = None

    # ==============================================================
    #                       FIT
    # ==============================================================
    def fit(self, df):
        """
        Learn statistics from the training dataset only.
        """
        # ---- Fit Age median ----
        self.age_median = df["Age"].median()

        # ---- Fit StandardScaler using a DataFrame (keeps feature names) ----
        age_filled = df[["Age"]].fillna(self.age_median)
        self.age_scaler = StandardScaler().fit(age_filled)

        return self


    PAP_MAP = {
        "NILM": 0,
        "ENDOMETRII-CELL":0.75,
        "ASCUS": 1,
        "LSIL": 2,
        "ASCH": 3,
        "HSIL": 4,
        "SCC": 4.5,
        "AGC": 5,
        "AGCNOS": 5.5,
        "AGC-NEOPLASI": 6,
        "ADENO-CARCINOMA": 6.5,
        "UNSATISFACTORY": 7}

    # ==============================================================
    #                   TRANSFORM HELPERS
    # ==============================================================

    # ---------- HPV ENCODING ----------
    @staticmethod
    def encode_hpv(x):
        if pd.isna(x) or str(x).strip() == "":
            return 0     # Missing → neutral
        x = str(x).strip().lower()
        if x == "No":
            return -2
        if x in ["Others"]:
            return -1
        if x in ["18", "hpv18"]:
            return 1
        if x in ["16", "hpv16"]:
            return 2
        return -1  # Unknown → treat as low risk

    def encode_hpv_from_columns(self, row):
        """
        row: a pandas Series with columns:
            '16', '18', 'Others'
        """
        # If all three are missing -> neutral (0)
        if row[["16", "18", "Others", "No"]].isna().all():
            return 0

        has_16 = bool(row.get("16", 0))
        has_18 = bool(row.get("18", 0))
        has_oth = bool(row.get("Others", 0))
        no_hpv = bool(row.get("No", 0))

        # No HPV at all (all zeros) -> -2
        if not has_16 and not has_18 and not has_oth:
            return -2

        codes = []
        if has_16:
            return 2.5
        if has_18:
            return 2
        if has_oth:
            return 1


    # ---------- PAP SMEAR ----------
    @staticmethod
    def encode_pap_category(x):
        if pd.isna(x) or str(x).strip() == "":
            return np.nan

        key = str(x)
        if key in Imputation.PAP_MAP:
            return Imputation.PAP_MAP[key]

        # Unknown category -> conservative choice (ASCUS)
        return 1

    # ---------- LUGOL ----------
    @staticmethod
    def encode_lugol_category(x):
        if pd.isna(x) or str(x).strip() == "":
            return 0
        x = str(x).strip().lower()
        if x == "negative":
            return -1
        if x == "positive":
            return 1
        return 0  # unknown → treat as negative

    # ==============================================================
    #                     TRANSFORM
    # ==============================================================
    def transform(self, df):
        """
        Apply imputations + encodings to any dataframe.
        Requires .fit() to have been called first.
        Returns a NEW dataframe (does not modify original df).
        """
        df = df.copy()

        # ========================================================
        # AGE
        # ========================================================
        df["Age"] = df["Age"].fillna(self.age_median)
        df["Age"] = self.age_scaler.transform(df[["Age"]])

        # ========================================================
        # HPV ENCODING
        # ========================================================
        df["HPV_encoded"] = df.apply(self.encode_hpv_from_columns, axis=1)


        # ========================================================
        # PAP SMEAR
        # ========================================================
        df["Pap_level_raw"] = df["Pop"].apply(self.encode_pap_category)
        df["Pap_missing"] = df["Pap_level_raw"].isna().astype(int)
        df["Pap_level"] = df["Pap_level_raw"].fillna(0)
        df.drop(columns=["Pap_level_raw"], inplace=True)

        # ========================================================
        # LUGOL
        # ========================================================
        df["Lugol"] = df["Logul"].apply(self.encode_lugol_category)
        # df["Lugol_missing"] = df["Lugol_raw"].isna().astype(int)
        # df["Lugol_value"] = df["Lugol_raw"].fillna(0)
        # df.drop(columns=["Lugol_raw"], inplace=True)

        # ========================================================
        # Final selected columns (tabular features)
        # ========================================================
        final_cols = [
            "Age",
            "HPV_encoded",
            "Pap_level",
            "Pap_missing",
            "Lugol",
        ]

        return df[final_cols].astype(np.float32)

    # ==============================================================
    #                FIT + TRANSFORM CONVENIENCE
    # ==============================================================
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
