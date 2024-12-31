#!/usr/bin/env python

# @Time    : 2024/9/26 9:29
# @Author  : Yao Xuan
# @Email   : xuany0512@163.com
# @File    : Dataset.py

from torch.utils.data import Dataset
import pandas as pd
import torch


class CL_CellVector(Dataset):
    def __init__(self, data_1, data_2, device='cuda'):
        if isinstance(data_1, str):
            self.data_1 = pd.read_csv(data_1, index_col=0)
        else:
            self.data_1 = data_1
        if isinstance(data_2, str):
            self.data_2 = pd.read_csv(data_2, index_col=0)
        else:
            self.data_2 = data_2

        self.device = device
        if len(self.data_1) != len(self.data_2):
            raise Exception("Length of datas is different!")
        if not all(self.data_1.index == self.data_2.index):
            raise Exception("Mismatching index!")

    def __len__(self):
        return len(self.data_1)

    def __getitem__(self, idx):
        cell = self.data_1.index[idx]
        features_1 = torch.tensor(self.data_1.iloc[idx].tolist(), dtype=torch.float, device=self.device)
        features_2 = torch.tensor(self.data_2.iloc[idx].tolist(), dtype=torch.float, device=self.device)

        return cell, features_1, features_2

    def get_dim(self):
        return (self.data_1.shape[1])


class CellVector(Dataset):
    def __init__(self, data, device='cuda'):
        if isinstance(data, str):
            self.data = pd.read_csv(data, index_col=0)
        else:
            self.data = data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cell = self.data.index[idx]
        feature = torch.tensor(self.data.iloc[idx].tolist(), dtype=torch.float, device=self.device)
        return cell, feature

    def get_dim(self):
        return (self.data.shape[1])
