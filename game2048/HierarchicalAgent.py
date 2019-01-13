from .agents import Agent
from utils import try_to_move, get_train_data, conv_to_onehot, ReplayMemory, Transition
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim
from model import nn2048, nn2048_2, nn2048_3
from .expectimax import board_to_move

DEFAULT_PATH0 = 'model3_dict0.pkl'
DEFAULT_PATH1 = 'model3_dict1.pkl'
BETA = 0.5
learning_rate = 5e-5
THRESHOLD = 0.5


class HierarchicalAgent(Agent):
    def __init__(self, game, display=None, train=True, load_data=False, path=None):
        super().__init__(game, display)
        self.train = train
        self.statistics = {2 ** i: 0 for i in range(1, 16)}
        self.threshold = THRESHOLD
        self.beta = BETA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step_counter = 0
        self.error_counter = 0
        self.diff_counter = 0
        if self.train:

            self.teacher = board_to_move

            if load_data:
                if path[0] is None:
                    path[0] = DEFAULT_PATH0
                else:
                    pass

                if path[1] is None:
                    path[1] = DEFAULT_PATH1
                else:
                    pass

                try:
                    self.net0 = nn2048_3().to(self.device)
                    self.net0.load_state_dict(torch.load(path[0], map_location=self.device))
                except FileNotFoundError:
                    print('No model loaded! Create new model')
                    self.net0 = nn2048_3().to(self.device)

                try:
                    self.net1 = nn2048_3().to(self.device)
                    self.net1.load_state_dict(torch.load(path[1], map_location=self.device))
                except FileNotFoundError:
                    print('No model loaded! Create new model')
                    self.net1 = nn2048_3().to(self.device)

            else:
                self.net0 = nn2048_3().to(self.device)
                self.net1 = nn2048_3().to(self.device)

            self.criterion0 = torch.nn.CrossEntropyLoss()
            self.criterion1 = torch.nn.CrossEntropyLoss()

            self.optimizer0 = torch.optim.Adam(self.net0.parameters(), lr=learning_rate)  # configure.learning_rate)
            self.optimizer1 = torch.optim.Adam(self.net1.parameters(), lr=learning_rate)

        else:
            ''' test without train '''
            try:
                if path[0] is None:
                    path[0] = DEFAULT_PATH0

                self.net0 = nn2048_3().to(self.device)
                self.net0.load_state_dict(torch.load(path[0], map_location=self.device))
                self.net0.eval()
            except FileNotFoundError:
                print('No model loaded!')
                self.net0 = nn2048_3().to(self.device)

            try:
                if path[1] is None:
                    path[1] = DEFAULT_PATH1

                self.net1 = nn2048_3().to(self.device)
                self.net1.load_state_dict(torch.load(path[1], map_location=self.device))
                self.net1.eval()
            except FileNotFoundError:
                print('No model loaded!')
                self.net1 = nn2048_3().to(self.device)

    def step(self):
        board = self.game.board
        oh_board = conv_to_onehot(board)
        self.step_counter += 1

        if self.train:
            target_direction = self.teacher(board)
            self.train_net(board, target_direction)

            if np.random.rand() > self.threshold or self.game.score < 512:
                direction = self.net0.predict(torch.Tensor(oh_board.reshape(1, *oh_board.shape)).to(self.device).float())

                if direction != target_direction:
                    self.error_counter += 1
            else:
                # f0 = self.net0.forward(torch.Tensor(oh_board.reshape(1, *oh_board.shape)).to(self.device).float())
                # f1 = self.net1.forward(torch.Tensor(oh_board.reshape(1, *oh_board.shape)).to(self.device).float())
                #
                # f = f0 + self.beta * f1
                # direction = torch.argmax(f, dim=1)
                direction = self.net1.predict(torch.Tensor(oh_board.reshape(1, *oh_board.shape)).to(self.device).float())

        else:
            """
                Only test without train
            """
            if self.game.score < 512:
                direction = self.net0.predict(torch.Tensor(oh_board.reshape(1, *oh_board.shape)).to(self.device).float())
            else:
                direction = self.net1.predict(torch.Tensor(oh_board.reshape(1, *oh_board.shape)).to(self.device).float())

        return direction

    def train_net(self, board, target_direction):
        train_data, train_targets = get_train_data(board, target_direction)

        train_data = torch.Tensor(train_data).to(self.device).float()
        train_targets = torch.Tensor(train_targets).to(self.device).long().squeeze(1)  #

        # if self.game.score <= 2048:
        #     y0 = self.net0.forward(train_data)
        #     loss0 = self.criterion0(y0, train_targets)
        #
        #     self.optimizer0.zero_grad()
        #     loss0.backward()
        #     self.optimizer0.step()

        if self.game.score >= 512:
            y1 = self.net1.forward(train_data)
            loss1 = self.criterion1(y1, train_targets)

            self.optimizer1.zero_grad()
            loss1.backward()
            self.optimizer1.step()

    def play(self, max_iter=np.inf, verbose=False):
        super(HierarchicalAgent, self).play(max_iter=max_iter, verbose=verbose)
        self.statistics[self.game.score] += 1

    def new_game(self, game):
        self.game = game


DEFAULT_TEST_PATH0 = 'model3_dict0.pkl'
DEFAULT_TEST_PATH1 = 'model3_dict1.pkl'


class TestAgent(Agent):
    def __init__(self, game, display=None):
        super().__init__(game, display)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net0 = nn2048_3().to(self.device)
        self.net1 = nn2048_3().to(self.device)
        self.net0.load_state_dict(torch.load(DEFAULT_TEST_PATH0, map_location=self.device))
        self.net0.eval()
        self.net1.load_state_dict(torch.load(DEFAULT_TEST_PATH1, map_location=self.device))
        self.net1.eval()

    def step(self):
        board = self.game.board
        oh_board = conv_to_onehot(board)
        if self.game.score < 512:
            direction = self.net0.predict(torch.Tensor(oh_board.reshape(1, *oh_board.shape)).to(self.device).float())
        else:
            direction = self.net1.predict(torch.Tensor(oh_board.reshape(1, *oh_board.shape)).to(self.device).float())
        return direction
