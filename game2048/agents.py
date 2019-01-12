import numpy as np


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction


class GenerateAgent(Agent):

    # counter = -1
    buffer = []
    stepCounter = 0
    STEPS_TO_SAVE = 500

    def __init__(self, game, display=None, steps_to_save=2000, direction='./'):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move
        # GenerateAgent.counter += 1
        self.fileName = direction + 'data_'
        GenerateAgent.STEPS_TO_SAVE = steps_to_save

    def step(self):
        board = self.game.board
        direction = self.search_func(board)

        GenerateAgent.stepCounter += 1

        if GenerateAgent.buffer == []:
            GenerateAgent.buffer = np.append(np.log2(board).flatten(), direction).reshape([1, 17]).astype(np.uint8)
        else:
            GenerateAgent.buffer = np.append(GenerateAgent.buffer,
                                             np.append(np.log2(board).flatten(), direction).reshape([1, 17]).astype(np.uint8),
                                             axis=0)

        if GenerateAgent.stepCounter % GenerateAgent.STEPS_TO_SAVE == 0:
            np.savetxt(self.fileName + '{0:d}.csv'.format(int(self.stepCounter / self.STEPS_TO_SAVE)),
                       self.buffer, fmt='%d')
            GenerateAgent.buffer = []

        return direction










