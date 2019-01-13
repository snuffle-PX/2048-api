# 2048-api
A 2048 game api for training supervised learning (imitation learning) or reinforcement learning agents

# Code structure
* [`game2048/`](game2048/): the main package.
    * [`game.py`](game2048/game.py): the core 2048 `Game` class.
    * [`agents.py`](game2048/agents.py): the `Agent` class with instances.
    * [`displays.py`](game2048/displays.py): the `Display` class with instances, to show the `Game` state.
    * [`expectimax/`](game2048/expectimax): a powerful ExpectiMax agent by [here](https://github.com/nneonneo/2048-ai).
    * [`my_agents.py`](game2048/my_agents.py): several agents I defined, for test, you should use TrainAgent with train.py or use TestAgent with evaluate.py.
    * [`HierarchicalAgent.py`](game2048/HierarchicalAgent.py): 
* [`model/`](model/): the package contains the models I uesd, and the nn2048_3 model is the model works best.
* [`utils/`](utils/): some function used in this project.
    * [`memory.py`](utils/memory.py): contains a buffer for train.
    * [`move.py`](utils/move.py): this is prepared for Reinforece Learning, unfortunately, I didn't work it out.
    * [`onehot.py`](utils/onehot.py): convert board to one hot style with 16 or 12 channels.
    * [`rot_invariance.py`](utils/rot_invariance.py): get train data through rotation and flip.
* [`log/`](log/): this is file for model visualize.
* [`explore.ipynb`](explore.ipynb): introduce how to use the `Agent`, `Display` and `Game`.
* [`static/`](static/): frontend assets (based on Vue.js) for web app.
* [`webapp.py`](webapp.py): run the web app (backend) demo.
* [`evaluate.py`](evaluate.py): evaluate your self-defined agent.
* [`train.py`](train.py): run to train model.
* [`test.py`](test.py): run to test model.
* [`model3_dict_01_11.pkl`](model3_dict_01_11.pkl): pre-trained network
* [`visualize.py`](visualize.py): run this to get the model graph

# Requirements
* code only tested on linux system (ubuntu 16.04)
* Python 3 (Anaconda 3.6.3 specifically) with numpy and flask

# To define your own agents
```python
from game2048.agents import Agent

class YourOwnAgent(Agent):

    def step(self):
        '''To define the agent's 1-step behavior given the `game`.
        You can find more instance in [`agents.py`](game2048/agents.py).
        
        :return direction: 0: left, 1: down, 2: right, 3: up
        '''
        direction = some_function(self.game)
        return direction

```

# To compile the pre-defined ExpectiMax agent

```bash
cd game2048/expectimax
bash configure
make
```

# To run the web app
```bash
python webapp.py
```
![demo](preview2048.gif)

# LICENSE

The code is under Apache-2.0 License.

# For EE369 students from SJTU only
Please read [here](EE369.md).

# For TA or anyone who want to run this project

### Run and test

You should compile expectedMax Agent if you want to use test.py or train.py, to just use evaluate.py, you can comment the line 9  `from .expectimax import board_to_move` in `game2048/my_agent.py`, and comment from line 20 to line 299. 

### Visualize

Requirements: `tensorflow`, ` tensorboardX`

Run `visualize.py` to get the log file, then run `tensorboard --logdir=./log` to see the model.



