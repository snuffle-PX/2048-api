from game2048.agents import ExpectiMaxAgent
from game2048.game import Game
from game2048.my_agents import TrainAgent, TrainAgent2, TrainAgent_12
from game2048.displays import Display
import torch

GAME_SIZE = 4
SCORE_TO_WIN = 2048
N_TESTS = 10

game = Game(GAME_SIZE, SCORE_TO_WIN)
agent = TrainAgent2(game, display=Display(), load_data=True, train=False,
        path='model3_dict_01_11.pkl')
for _ in range(50):
    agent.play(verbose=False)
    agent.new_game(game=Game(GAME_SIZE, SCORE_TO_WIN))
    print('Average time per step: {}'.format(agent.t / agent.step_counter))

print(agent.statistics)
print('steps: {}, errors: {}, differents: {}'.format(agent.step_counter,
    agent.error_counter, agent.diff_counter))
# torch.save(agent.net.state_dict(), 'model.pkl')

