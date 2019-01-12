from game2048.game import Game
from game2048.my_agents import TrainAgent
from game2048.HierarchicalAgent import HierarchicalAgent
from game2048.displays import Display
import torch

GAME_SIZE = 4
SCORE_TO_WIN = 4096
N_TESTS = 10
PATH = ('model3_dict_01_11.pkl', 'model3_dict1.pkl')
SAVE_PATH = ('model3_dict0.pkl', 'model3_dict1.pkl')

game = Game(GAME_SIZE, SCORE_TO_WIN, random=False)
agent = HierarchicalAgent(game, display=Display(), load_data=True, path=PATH)
for i in range(2000):
    agent.play(verbose=False)
    print("Game: {} Score: {}".format(i, agent.game.score))
    agent.new_game(game=Game(GAME_SIZE, SCORE_TO_WIN, random=False))
    agent.threshold *= 0.99
    # print("Steps: {} Errors: {}".format(agent.step_counter, agent.error_counter))
    if (i+1) % 100 == 0:
        torch.save(agent.net0.state_dict(), SAVE_PATH[0])
        torch.save(agent.net1.state_dict(), SAVE_PATH[1])

print(agent.statistics)
torch.save(agent.net0.state_dict(), SAVE_PATH[0])
torch.save(agent.net1.state_dict(), SAVE_PATH[1])
