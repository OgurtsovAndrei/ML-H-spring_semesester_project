import pickle
import torch

import random
import numpy as np

import torch
import torch.nn.functional as F

def generate_data(num_data, dim, num_sum, fn, support):
    data = torch.FloatTensor(dim).uniform_(*support).unsqueeze_(1)
    X, y = [], []
    for i in range(num_data):
        idx_a = random.sample(range(dim), num_sum)
        idx_b = random.sample([x for x in range(dim) if x not in idx_a], num_sum)
        a, b = data[idx_a].sum(), data[idx_b].sum()
        X.append([a, b])
        y.append(fn(a, b))
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).unsqueeze_(1)
    indices = list(range(num_data))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    return X, y