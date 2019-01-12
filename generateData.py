from game2048.game import Game
from game2048.agents import GenerateAgent

size = 4
score_to_win = 2048
epoch = 5000
for i in range(epoch):
    game = Game(size, score_to_win)
    agent = GenerateAgent(game, display=None, direction='./dataSet/')
    agent.play(verbose=True)
