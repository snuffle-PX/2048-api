from .agents import Agent
from utils import try_to_move, get_train_data, conv_to_onehot, ReplayMemory, Transition, conv_to_onehot_12, get_train_data_12
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim
from model import nn2048, nn2048_2, nn2048_3, nn2048_4
from .expectimax import board_to_move
import time


BATCH_SIZE = 64
TARGET_UPDATE = 10
learning_rate = 1e-4
THRESHOLD = 0.5
DEFAULT_PATH = 'model_dict.pkl'


class TrainAgent(Agent):

    def __init__(self, game, display=None, train=True, load_data=False, path=None):
        super().__init__(game, display)
        self.train = train
        self.statistics = {2 ** i: 0 for i in range(1, 16)}
        self.threshold = THRESHOLD
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step_counter = 0
        self.error_counter = 0
        self.diff_counter = 0
        self.t = 0
        if self.train:

            self.teacher = board_to_move

            if load_data:
                if path is None:
                    path = DEFAULT_PATH
                else:
                    pass
                try:
                    self.net = nn2048_3().to(self.device)
                    self.net.load_state_dict(torch.load(path, map_location=self.device))
                except FileNotFoundError:
                    print('No model loaded! Create new model')
                    self.net = nn2048_3().to(self.device)
            else:
                self.net = nn2048_3().to(self.device)

            self.criterion = torch.nn.CrossEntropyLoss()

            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)  # configure.learning_rate)
            self.buffer = ReplayMemory(5 * BATCH_SIZE)

        else:
            try:
                if path is None:
                    path = DEFAULT_PATH

                self.net = nn2048_3().to(self.device)
                self.net.load_state_dict(torch.load(path, map_location=self.device))
                self.net.eval()
            except FileNotFoundError:
                print('No model loaded!')
                self.net = nn2048_3().to(self.device)

    def train_net(self, board, target_direction):
        # target_direction = self.teacher(board)
        train_data, train_targets = get_train_data(board, target_direction)
        self.buffer.push6(train_data, train_targets)
        transitions = self.buffer.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))
        train_data = torch.Tensor(batch.board).to(self.device).float()
        train_targets = torch.Tensor(batch.direction).to(self.device).long().squeeze(1)

        # train_data = torch.Tensor(train_data).to(self.device).float()
        # train_targets = torch.Tensor(train_targets).to(self.device).long().squeeze(1)  # squeeze() delete the dimention of 1

        y = self.net.forward(train_data)
        loss = self.criterion(y, train_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def step(self):
        start = time.time()
        board = self.game.board
        oh_board = conv_to_onehot(board)
        self.step_counter += 1

        if self.train:
            target_direction = self.teacher(board)

            self.train_net(board, target_direction)

            if np.random.rand() > self.threshold or self.game.score < 512 :
                direction = self.net.predict(torch.Tensor(oh_board.reshape(1, *oh_board.shape)).to(self.device).float())

                if direction != target_direction:
                    self.error_counter += 1
                # _, score = try_to_move(board, direction)
                # if score == -1:  # cannot move to the selected direction
                #     # direction = target_direction
                #     #print("score -1")
                #     self.error_counter += 1
            else:
                direction = target_direction

        else:
            """
                Only test without train
            """
            direction = self.net.predict(torch.Tensor(oh_board.reshape(1, *oh_board.shape)).to(self.device).float())
            _, score = try_to_move(board, direction)
            if score == -1:  # cannot move to the selected direction
                self.error_counter += 1

            if direction != board_to_move(board):
                self.diff_counter += 1
                # direction = board_to_move(board)
                # print("score -1")

        end = time.time()
        self.t += start - end
        return direction

    def play(self, max_iter=np.inf, verbose=False):
        super(TrainAgent, self).play(max_iter=max_iter, verbose=verbose)
        self.statistics[self.game.score] += 1

    def new_game(self, game):
        self.game = game


class TrainAgent2(TrainAgent):
    def train_net(self, board, target_direction):
        train_data, train_targets = get_train_data(board, target_direction)
        train_data = torch.Tensor(train_data).to(self.device).float()
        train_targets = torch.Tensor(train_targets).to(self.device).long().squeeze(1)  #

        y = self.net.forward(train_data)
        loss = self.criterion(y, train_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class RLAgent(Agent):

    def __init__(self, game, display=None, train=True, load_data=False, path=None):
        super().__init__(game, display)
        self.train = train
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam(betas=(0.5, 0.999))
        self.criterion = torch.nn.MSELoss(size_average=True)
        self.net = nn2048()
        self.last_board = None

    def step(self):
        board = self.game.board
        oh_board = conv_to_onehot(board)
        board_list = []
        for d in range(4):
            _, score = try_to_move(board, d)
            if score >= 0:
                board_list.append((d, score))

        if board_list:
            s = [self.net.predict(torch.Tensor(oh_board.reshape(1, *oh_board.shape)).to(self.device).float()) + score*2
                 for d, score in board_list]
            idx = np.argmax(s)
            value = np.max(s)
            direction = board_list[idx][1]
            if self.train and any(self.last_board):
                self.train_net(self.last_board, value)
                self.last_board = board
            return direction

    def train_net(self, last_board, value):
        train_data, _ = get_train_data(last_board, 0)
        train_targets = np.array([value]*6).reshape(-1, 1)

        train_data = torch.Tensor(train_data).to(self.device).float()
        train_targets = torch.Tensor(train_targets).to(self.device).long().squeeze(1)  #

        y = self.net.forward(train_data)
        loss = self.criterion(y, train_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class TrainAgent_12(Agent):

    def __init__(self, game, display=None, train=True, load_data=False, path=None):
        super().__init__(game, display)
        self.train = train
        self.statistics = {2 ** i: 0 for i in range(1, 16)}
        self.threshold = THRESHOLD
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step_counter = 0
        self.error_counter = 0
        self.diff_counter = 0
        self.t = 0
        if self.train:

            self.teacher = board_to_move

            if load_data:
                if path is None:
                    path = DEFAULT_PATH
                else:
                    pass
                try:
                    self.net = nn2048_4().to(self.device)
                    self.net.load_state_dict(torch.load(path, map_location=self.device))
                except FileNotFoundError:
                    print('No model loaded! Create new model')
                    self.net = nn2048_4().to(self.device)
            else:
                self.net = nn2048_4().to(self.device)

            self.criterion = torch.nn.CrossEntropyLoss()

            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)  # configure.learning_rate)
            # self.buffer = ReplayMemory(5 * BATCH_SIZE)

        else:
            try:
                if path is None:
                    path = DEFAULT_PATH

                self.net = nn2048_4().to(self.device)
                self.net.load_state_dict(torch.load(path, map_location=self.device))
                self.net.eval()
            except FileNotFoundError:
                print('No model loaded!')
                self.net = nn2048_4().to(self.device)

    def train_net(self, board, target_direction):
        train_data, train_targets = get_train_data_12(board, target_direction)
        train_data = torch.Tensor(train_data).to(self.device).float()
        train_targets = torch.Tensor(train_targets).to(self.device).long().squeeze(1)  #

        y = self.net.forward(train_data)
        loss = self.criterion(y, train_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def step(self):
        start = time.time()
        board = self.game.board
        oh_board = conv_to_onehot_12(board)
        self.step_counter += 1

        if self.train:
            target_direction = self.teacher(board)

            self.train_net(board, target_direction)

            if np.random.rand() > self.threshold or self.game.score < 512:
                direction = self.net.predict(torch.Tensor(oh_board.reshape(1, *oh_board.shape)).to(self.device).float())

                if direction != target_direction:
                    self.error_counter += 1

            else:
                direction = target_direction

        else:
            """
                Only test without train
            """
            direction = self.net.predict(torch.Tensor(oh_board.reshape(1, *oh_board.shape)).to(self.device).float())
            _, score = try_to_move(board, direction)
            if score == -1:  # cannot move to the selected direction
                self.error_counter += 1

            if direction != board_to_move(board):
                self.diff_counter += 1
                # direction = board_to_move(board)
                # print("score -1")

        end = time.time()
        self.t += start - end
        return direction

    def play(self, max_iter=np.inf, verbose=False):
        super(TrainAgent_12, self).play(max_iter=max_iter, verbose=verbose)
        self.statistics[self.game.score] += 1

    def new_game(self, game):
        self.game = game


DEFAULT_TEST_PATH = 'model3_dict_01_11.pkl'


class TestAgent(Agent):
    def __init__(self, game, display=None):
        super().__init__(game, display)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn2048_3().to(self.device)
        self.net.load_state_dict(torch.load(DEFAULT_TEST_PATH, map_location=self.device))
        self.net.eval()

    def step(self):

        board = self.game.board
        oh_board = conv_to_onehot(board)
        direction = self.net.predict(torch.Tensor(oh_board.reshape(1, *oh_board.shape)).to(self.device).float())
        direction = int(direction.data.numpy())
        return direction
