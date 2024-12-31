#!/usr/bin/env python

# @Time    : 2024/9/24 13:45
# @Author  : Yao Xuan
# @Email   : xuany0512@163.com
# @File    : Transform.py

import torch

class GaussianNoise(object):
    def __init__(self, mean=0.1, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        noise = torch.normal(self.mean, self.std, size=x.shape, device=x.device)
        output = x + noise
        output[output < 0] = 0
        return output


class UniformNoise(object):
    def __init__(self, low=-0.1, high=0.1):
        self.low = low
        self.high = high

    def __call__(self, x):
        noise = torch.empty_like(x, device=x.device).uniform_(self.low, self.high)
        output = x + noise
        output[output < 0] = 0
        return output



class RandomDropout(object):
    def __init__(self, dropout_rate=0.2):
        self.dropout_rate = dropout_rate

    def __call__(self, x):
        noise = torch.rand_like(x, device=x.device)
        x[noise < self.dropout_rate] = 0
        return x
