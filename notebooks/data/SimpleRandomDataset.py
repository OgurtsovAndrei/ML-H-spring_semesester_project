from torch.utils.data import Dataset
import torch
import os
import torch.random
import numpy as np
from src.tools import FloatBitsConverter


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

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[:, idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


class SimpleRandomBitDataset(Dataset):
    """Simplest random dataset"""

    def __init__(self, n, left_bound=-5, right_bound=5, transform=None):
        """

        :param left_bound:
        :param right_bound:
        """
        self.size = n
        self.transform = transform
        random_array = np.random.rand(n) * (right_bound - left_bound) + left_bound
        random_array = np.array([FloatBitsConverter.get_float_bits_in_memory(val) for val in random_array])
        random_array = random_array.reshape((len(random_array), 64)).astype('float64')
        self.data = np.array([random_array, random_array])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[:, idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    data = SimpleRandomBitDataset(100)
    print(data[20])
