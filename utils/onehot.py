import numpy as np


def conv_log_to_onehot(grid):
    one_hot = np.zeros((16,) + grid.shape)  # , dtype=np.int)
    for i in range(16):
        one_hot[i, grid == i] = 1
    return one_hot


def conv_to_onehot(grid):
    """

    :param grid: matrix with item equal to 0 or time of 2
    :return: 16*grid.shape one-hot format 3D matrix
    """
    one_hot = np.zeros((16,) + grid.shape)  # , dtype=np.int)
    one_hot[0, grid == 0] = 1
    for i in range(1,16):
        one_hot[i, grid == 2**i] = 1

    return one_hot


def flatten_onehot(one_hot):
    """

    :param one_hot: 16*grid.shape one-hot format 3D matrix
    :return: matrix with item equal to 0 or time of 2
    """
    grid = np.argmax(one_hot, axis=0)
    grid = 2 ** grid
    grid[grid == 1] = 0
    return grid
