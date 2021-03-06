"""
    author: 赵阳桁
    此文件中通过旋转，对称的方式对每个棋盘生成5个棋盘和响应的方向，作为一个batch交给网络训练
"""
import numpy as np
from .onehot import conv_to_onehot, conv_to_onehot_12


def get_train_data(board, direction):
    """
    direction:
                0: left
                1: down
                2: right
                3: up
    """
    train_data = np.zeros((6, 16) + board.shape)  # , dtype=np.int16)
    train_targets = np.zeros([6, 1])  # , dtype=np.int8)
    train_data[0, :, :, :] = conv_to_onehot(board)
    train_targets[0] = direction

    # generate 3 board with rotation
    for i in range(1, 4):
        train_data[i, :, :, :] = conv_to_onehot(np.rot90(board, i))
        train_targets[i] = (direction + i) % 4

    row_map = {0: 2, 1: 1, 2: 0, 3: 3}
    col_map = {0: 0, 1: 3, 2: 2, 3: 1}

    train_data[4, :, :, :] = conv_to_onehot(board[:, -1::-1])
    train_targets[4] = row_map[direction]

    train_data[5, :, :, :] = conv_to_onehot(board[-1::-1, :])
    train_targets[5] = col_map[direction]

    return train_data, train_targets


def get_train_data_12(board, direction):
    """
    direction:
                0: left
                1: down
                2: right
                3: up
    """
    train_data = np.zeros((6, 12) + board.shape)  # , dtype=np.int16)
    train_targets = np.zeros([6, 1])  # , dtype=np.int8)
    train_data[0, :, :, :] = conv_to_onehot_12(board)
    train_targets[0] = direction

    # generate 3 board with rotation
    for i in range(1, 4):
        train_data[i, :, :, :] = conv_to_onehot_12(np.rot90(board, i))
        train_targets[i] = (direction + i) % 4

    row_map = {0: 2, 1: 1, 2: 0, 3: 3}
    col_map = {0: 0, 1: 3, 2: 2, 3: 1}

    train_data[4, :, :, :] = conv_to_onehot_12(board[:, -1::-1])
    train_targets[4] = row_map[direction]

    train_data[5, :, :, :] = conv_to_onehot_12(board[-1::-1, :])
    train_targets[5] = col_map[direction]

    return train_data, train_targets
