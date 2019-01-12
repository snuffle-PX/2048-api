from game2048.agents import ExpectiMaxAgent
from game2048.game import Game
from game2048.my_agents import TrainAgent
from game2048.displays import Display
import torch

GAME_SIZE = 4
SCORE_TO_WIN = 4096
N_TESTS = 10
PATH = 'model_dict.pkl'
SAVE_PATH = 'model_dict_new.pkl'

game = Game(GAME_SIZE, SCORE_TO_WIN, random=False)
agent = TrainAgent(game, display=Display(), load_data=True, path=PATH)
for i in range(2000):
    agent.play(verbose=False)
    print("Game: {} Score: {}".format(i, agent.game.score))
    agent.new_game(game=Game(GAME_SIZE, SCORE_TO_WIN, random=False))
    agent.threshold *= 0.99
    print("Steps: {} Errors: {}".format(agent.step_counter,agent.error_counter))
    if (i+1)%100 == 0:
        torch.save(agent.net.state_dict(),SAVE_PATH)

print(agent.statistics)
torch.save(agent.net.state_dict(), SAVE_PATH)
# torch.save(agent.net, 'model.pkl')
