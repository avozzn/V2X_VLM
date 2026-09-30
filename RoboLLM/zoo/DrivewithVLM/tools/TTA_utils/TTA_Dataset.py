from torch.utils.data import Dataset
import torch

class TTADataset(Dataset):
    def __init__(self, data):
        self.data = data  

    def __len__(self):
        return 1 

    def __getitem__(self, idx):
        sample = self.data
        processed_sample = {}
        for key, value in sample.items():
                processed_sample[key] = value    
        return processed_sample



