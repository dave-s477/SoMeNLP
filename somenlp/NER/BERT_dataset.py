import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

class BERTDataset(Dataset):
    """PyTorch Dataset for BERT data
    """
    def __init__(self, ids, tags, masks, transforms=None):
        self.ids = ids
        self.tags = tags
        self.masks = masks
        self.transforms = transforms
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {
            'ids': self.ids[idx], 
            'masks': self.masks[idx],
            'tags': self.tags[idx]
        }

        if self.transforms:
            sample = self.transforms(sample)
        return sample
    