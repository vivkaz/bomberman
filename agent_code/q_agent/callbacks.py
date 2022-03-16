import os
import pickle
import random

import numpy as np
from sklearn import neighbors


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
N_STATES = 825
M_ACTIONS = len(ACTIONS)
ALPHA = 0.1
GAMMA = 0.6
epsilon = 0.1


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        q_table = np.zeros([N_STATES, M_ACTIONS])
        self.model = q_table
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    if self.train and random.uniform(0, 1) < epsilon: # random.random()
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1]) # Explore action space

    self.logger.debug("Querying model for action.")
    #return np.random.choice(ACTIONS, p=self.model)
    state = state_to_features(game_state)
    action = np.argmax(self.model[state]) # Exploit learned values
    
    return ACTIONS[action]



def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    
    position = game_state['self'][3] # (x,y) coordinate on the field
    if True:
        position_value = 0
    channels.append(position_value)
    sub = [(1,0), (-1,0), (0,1), (0,-1)]
    neighbors = [np.subtract(position, i) for i in sub]
    features_1_4 = game_state['field'][neighbors] # Its entries are 1 for crates, −1 for stone walls and 0 for free tiles

    channels.append(features_1_4)

    # TODO: value of current field (position), more values for features_1_4

    if len(game_state['coins']) > 0: # TODO: check if the coins are visible, near the agent
        mode = 0 # collect coins
    elif len(game_state['others']) > 0:
        mode = 2 # kill opponents
    else:
        mode = 1 # destroy crates

    channels.append(mode)

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    
    # and return them as a vector
    return stacked_channels.reshape(-1)

