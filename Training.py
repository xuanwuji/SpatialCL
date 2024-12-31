#!/usr/bin/env python

# @Time    : 2024/9/26 10:25
# @Author  : Yao Xuan
# @Email   : xuany0512@163.com
# @File    : Training.py

import Trainer
import torch
from train_util import plot_loss
import os

epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = Trainer.CL_Trainer(data_1=r"Z:\Work\Post_COVID19\workflow\dataset\test.csv",
                             data_2=r"Z:\Work\Post_COVID19\workflow\dataset\test.csv",
                             device=device,)
all_train_loss = []
save_path = './output.4'
if not os.path.exists(save_path):
    os.mkdir(save_path)

for epoch in range(1, epochs):
    loss, model, optimizer = trainer.train()
    all_train_loss.append(loss)
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    torch.save(model.state_dict(), save_path + '/save_model.pt')
    plot_loss(max_epochs=len(all_train_loss), data=all_train_loss, color='red', label='Train Loss',
              save_path=save_path + "/train_loss")
