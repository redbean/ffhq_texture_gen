import os, sys

import numpy as np
from torch.utils.data import Dataset



class NumpyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.data_files = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_files[idx])
        data = np.load(data_path)
        
        if self.transform:
            data = self.transform(data)
        
        return data

class CombinedDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_files = sorted(os.listdir(input_dir))
        self.target_files = sorted(os.listdir(target_dir))
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform

    def __len__(self):
        return min(len(self.input_files), len(self.target_files))

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])
        
        input_data = np.load(input_path)
        target_data = np.load(target_path)
        
        if self.transform:
            input_data = self.transform(input_data)
            target_data = self.transform(target_data)
        
        return input_data, target_data
