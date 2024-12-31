#!/usr/bin/env python

# @Time    : 2024/9/26 9:56
# @Author  : Yao Xuan
# @Email   : xuany0512@163.com
# @File    : Trainer.py

import torch
from Model import Model2
# 第三方模块，定义了许多对比损失函数:https://kevinmusgrave.github.io/pytorch-metric-learning/
from pytorch_metric_learning.losses import NTXentLoss
import tqdm
from torch.utils.data import DataLoader
import Dataset


class CL_Trainer(object):
    def __init__(self, data_1, data_2, bs=128, lr=0.001, t=0.5,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.bs = bs
        self.lr = lr
        self.device = device

        self.t = t
        self.d_model = 1024
        self.d_proj = 512
        self.n_layer = 2
        self.data_1 = data_1
        self.data_2 = data_2

        # 训练配置初始化
        self.dataset = Dataset.CL_CellVector(self.data_1, self.data_2, device=device)
        self.data_loader = DataLoader(self.dataset, batch_size=bs, shuffle=True, drop_last=False)
        self.loss_func = NTXentLoss(temperature=self.t)  # This is also known as InfoNCE
        self.model = Model2(self.dataset.get_dim(), self.d_model, self.d_proj, self.n_layer, device=self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        print("========Model Init==========")
        print(self.model)
        print('============================')

    def set_model(self, d_model, d_proj, n_layer):
        self.model = Model2(self.dataset.get_dim(), d_model=d_model, out_dim=d_proj, n_layer=n_layer,
                            device=self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        print("========Model Change==========")
        print(self.model)
        print('============================')
        return "Model has been updated successfully!"

    def train(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm.tqdm(self.data_loader)
        for step, (_, cell_ts_features_1, cell_ts_features_2) in enumerate(pbar):
            self.optimizer.zero_grad()
            # Get data representations
            x1 = self.model(cell_ts_features_1)
            x2 = self.model(cell_ts_features_2)
            # Prepare for loss
            embeddings = torch.cat((x1, x2))
            # print(embeddings.size())
            # The same index corresponds to a positive pair
            indices = torch.arange(0, x1.size(0), device=x1.device)
            labels = torch.cat((indices, indices))
            loss = self.loss_func(embeddings, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_description("Model training")
        total_loss = total_loss / (step + 1)
        return total_loss, self.model, self.optimizer
