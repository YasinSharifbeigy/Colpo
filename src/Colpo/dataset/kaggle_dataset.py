# src/datasets/kaggle_dataset.py
import os
from PIL import Image
import torch
from .base import BaseDataset

class KaggleImageDataset(BaseDataset):
    def __init__(self, root, split, transform=None):
        """
        root/
          Train/Normal
          Train/Abnormal
          ...
        """
        self.transform = transform
        self.images = []
        self.labels = []

        split_dir = os.path.join(root, split)
        class_to_idx = {"Normal": 0, "Abnormal": 1}

        for cls, label in class_to_idx.items():
            cls_dir = os.path.join(split_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    img = Image.open(os.path.join(cls_dir, fname)).convert("RGB")
                    if transform:
                        img = transform(img)
                    self.images.append(img)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
