from torch.utils.data import Dataset
import torch
import os
import torch.random


class SimpleRandomDataset(Dataset):
    """Simplest random dataset"""

    def __init__(self, n, left_bound=-5, right_bound=5, transform=None):
        """

        :param left_bound:
        :param right_bound:
        """
        self.transform = transform
        random_array = torch.rand(n) * (right_bound - left_bound) + left_bound
        self.data = pd.DataFrame({
            'X': random_array,
            'y': random_array,
        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

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

        sample = self.data.iloc[idx].T.to_numpy()

        print(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    data = SimpleRandomDataset(100)
    print(data[:20].T.to_numpy())

import pandas as pd
