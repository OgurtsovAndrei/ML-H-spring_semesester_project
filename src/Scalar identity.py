#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[3]:


import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# In[46]:


activation_fun = nn.ReLU()
layer_sizes = [1, 8, 8, 8, 1]


class SamenessModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.run_counter = 0

        layers_list = []
        for i in range(len(layer_sizes) - 1):
            layers_list.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i != len(layer_sizes) - 2:
                layers_list.append(activation_fun)

        self.l1 = nn.Sequential(*layers_list)
        # self.layers = layers_list
        print(self)

        for layer in self.l1:
            if isinstance(layer, nn.Linear):
                layer.weight = nn.Parameter(layer.weight.double())
                layer.bias = nn.Parameter(layer.bias.double())

    def forward(self, x):
        self.run_counter += 1
        return self.l1(x.double())    # <--- Вылетает ошибка тут


# class Decoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
#
#     def forward(self, x):
#         return self.l1(x)


class SamenessAutoEncoder(pl.LightningModule):
    def __init__(self, encoder: SamenessModule):
        super().__init__()
        self.encoder = encoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # print("\n" + "-" * 20)
        # print(f"batch: {batch.shape}")
        # print(f"batch: {batch}")
        # print("-" * 20)
        batch = batch[:, 0, :]
        x, y = batch.reshape(2, len(batch), 1)
        x_hat = self.encoder(x)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5 * 1e-3)
        return optimizer


# In[47]:


from torch.utils.data import Dataset
import torch
import os
import torch.random
import pandas as pd


# In[48]:


# from notebooks.data.SimpleRandomDataset import SimpleRandomDataset


# In[49]:


class SimpleRandomDataset(Dataset):
    """Simplest random dataset"""

    def __init__(self, n, left_bound=-5, right_bound=5, transform=None):
        """

        :param left_bound:
        :param right_bound:
        """
        self.size = n
        self.transform = transform
        random_array = np.random.rand(n) * (right_bound - left_bound) + left_bound
        random_array = random_array.reshape((len(random_array), 1))
        self.data = np.array([random_array, random_array])
        # self.data = pd.DataFrame({
        #     'X': random_array,
        #     'y': random_array,
        # })


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # print(idx)
        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}
        #
        # if self.transform:
        #     sample = self.transform(sample)
        #
        # return sample
        # pass
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # if isinstance(idx, int):
        #     idx = [idx]

        # X = self.data[idx]
        # y = self.data[idx]
        # sample = {"X": X, "y": y}

        # sample = self.data.iloc[idx].to_numpy()
        sample = self.data[:, idx]
        # sample = np.array([X, y])

        # FIXME:
        # if idx is int:
        #     sample = self.data.iloc[idx].to_list()
        #     sample = [[sample[0]], [sample[1]]]
        # else:
        # sample = self.data.iloc[idx].to_numpy().T

        # print("\n" + "-" * 20)
        # print(f"sample: {sample}")
        # print("-" * 20)
        if self.transform:
            sample = self.transform(sample)
        # print(f"sample: {sample}")
        # print("-" * 20)
        return sample


# In[51]:


dataset = SimpleRandomDataset(50000, transform=transforms.ToTensor())
train_loader = DataLoader(dataset, batch_size=10000)

# model
autoencoder = SamenessAutoEncoder(SamenessModule())

# train model
trainer = pl.Trainer(max_epochs=250)
# trainer.fit(model=autoencoder, train_dataloaders=train_loader)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)


# In[24]:


from tqdm import tqdm


# In[55]:


test_dataset = SimpleRandomDataset(1000, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=10)

sum_dif = 0
with torch.no_grad():
    # for batch in tqdm(test_loader):
    for index, batch in tqdm(enumerate(test_loader)):
        batch = batch[:, 0, :]
        x, y = batch.reshape(2, 10, 1)
        reconstructed_x = autoencoder.encoder(x)
        if index == 0:
            print("Original x:", x)
            print("Reconstructed x:", reconstructed_x)
        sum_dif += sum((x - reconstructed_x)**2)
print(f"square sum: {(sum_dif / (index + 1) / 10).__float__()}")

# In[32]:


autoencoder.encoder.run_counter


# In[ ]:


n = 1000
s = 0
c = 0
with torch.no_grad():
    for val in test_dataset:
        c += 1
        s += autoencoder.encoder(val)
print(f"avarage diff: {s/c}")


# In[ ]:


autoencoder.encoder.run_counter


# In[ ]:


val = torch.tensor([[1],[2],[4],[5]])
autoencoder.encoder(val)


x = np.linspace(-20, 20, 1000)
x = x.reshape(len(x), 1)
x = torch.tensor(x)

y = autoencoder.encoder(x)

x = x.reshape(len(x)).detach().numpy()
y = y.reshape(len(y)).detach().numpy()
differ = (x - y)**2
print(x.shape)
print(y.shape)
print(differ.shape)

import plotly.express as px
fig = px.line(x=x, y=y)
fig.show()

fig = px.line(x=x, y=differ)
fig.show()