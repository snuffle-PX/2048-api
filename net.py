import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


def get_one_hot(mat):
    one_hot_mat = np.zeros((4, 4, 16), dtype=np.int)
    for i in range(16):
        one_hot_mat[mat == i, i] = 1
    return one_hot_mat


class Game2048DataSet(Dataset):
    filename = 'dataSet/data_'

    def __init__(self, r):
        self.r = r
        self.num = 1
        self.file = Game2048DataSet.filename + '{0:d}'.format(self.num) + '.csv'
        self.data = np.loadtxt(self.filename, dtype=np.int)

    def __getitem__(self, item):
        num = np.floor(item, 10000)
        if num != self.num:
            self.num = num
            self.file = Game2048DataSet.filename + '{0:d}'.format(self.num) + '.csv'
            self.data = np.loadtxt(self.filename, dtype=np.int)

        grid = np.reshape(self.data[item % 10000, 0:17], [4, 4])
        target = self.data[item % 10000, 17]
        return get_one_hot(grid), target

    def __len__(self):
        return 10000


