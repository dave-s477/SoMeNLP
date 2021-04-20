import torch
import numpy as np
import pandas as pd

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset

class LSTMDataset(Dataset):
    """PyTorch dataset for LSTM input data
    """
    def __init__(self, characters, ids, tags, features, character2idx, padding, max_word_length, max_sent_length, transforms=None):
        self.characters = characters
        self.ids = ids
        self.tags = tags
        self.transforms = transforms
        self.character2idx = character2idx
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.padding = padding
        self.features = features
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.max_sent_length > 0:
            sample = {
                'ids': torch.tensor(self.ids[idx][:self.max_sent_length]), 
                'features': torch.tensor(self.features[idx][:self.max_sent_length]) if self.features is not None else None, 
                'tags': torch.tensor(self.tags[idx][:self.max_sent_length])}
        else: 
            sample = {
                'ids': torch.tensor(self.ids[idx]), 
                'features': torch.tensor(self.features[idx]) if self.features is not None else None,
                'tags': torch.tensor(self.tags[idx])}

        if self.max_word_length < 0:
            sample['characters'] = pad_sequence([torch.tensor(x) for x in self.characters[idx]], batch_first=True, padding_value=self.character2idx[self.padding])
        else:
            sample['characters'] = pad_sequence([torch.tensor(x[:self.max_word_length]) for x in self.characters[idx]], batch_first=True, padding_value=self.character2idx[self.padding])

        if self.transforms:
            sample = self.transforms(sample)
        return sample
    