"""
    author: 赵阳桁
    ReplayMemory 是一个用于缓存的buffer
"""
from collections import namedtuple
import random

Transition = namedtuple("Transition", ("board", "direction"))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.p = 0

    def push6(self, boards, directions):
        for b, d in zip(boards, directions):
            self.push(b, d)

    def push(self, board, direction):  # 3D matrix
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.p] = Transition(board, direction)
        self.p = (self.p + 1) % self.capacity

    def sample(self, batch_size):
        size = len(self.memory)
        if size < batch_size:
            return random.sample(self.memory, size)
        else:
            return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
