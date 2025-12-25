# src/datasets/merge.py
from torch.utils.data import ConcatDataset

def merge_datasets(datasets):
    return ConcatDataset(datasets)
