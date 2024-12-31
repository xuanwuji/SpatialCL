#!/usr/bin/env python

# @Time    : 2024/9/23 16:31
# @Author  : Yao Xuan
# @Email   : xuany0512@163.com
# @File    : Model.py

import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim, d_model, out_dim, device):
        super(Model, self).__init__()
        self.embedding_extractor = nn.Sequential(
            nn.Linear(input_dim, d_model, device=device),
            nn.SELU(),
            nn.Linear(d_model, d_model * 2, device=device),
            nn.SELU(),
            nn.Linear(d_model * 2, d_model, device=device),
            nn.SELU()
        )

        self.projector = nn.Sequential(
            nn.Linear(d_model, out_dim, device=device),
            nn.SELU()
        )

    def forward(self, x):
        x = self.embedding_extractor(x)
        x = self.projector(x)
        return x


def create_block(d_model, device):
    block = nn.Sequential(
        nn.Linear(d_model, d_model * 2, device=device),
        nn.ELU(),
        nn.Linear(d_model * 2, d_model, device=device),
        nn.ELU(),
        nn.BatchNorm1d(d_model, device=device)
    )
    return block


class Model2(nn.Module):
    def __init__(self, input_dim, d_model, out_dim, n_layer, device):
        super(Model2, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model, device=device),
            nn.ELU(),
            nn.BatchNorm1d(d_model, device=device)
        )
        self.embedding_extractor = nn.ModuleList(
            [create_block(d_model, device) for i in range(n_layer)]
        )

        self.projector = nn.Sequential(
            nn.Linear(d_model, out_dim, device=device),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.embedding(x)
        for embedding_extractor_layer in self.embedding_extractor:
            x = embedding_extractor_layer(x)
        x = self.projector(x)
        return x
