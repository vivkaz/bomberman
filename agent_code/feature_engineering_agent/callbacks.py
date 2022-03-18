import os
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import deque
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']





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
    # initialize the model and save as self.model_train
    # if mode is train the model self.model_train becomes self.model

    # number of outputs of the model
    self.n_outputs = 5
    self.inputs_shape = (17, 17, 2)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(2, 3, activation="relu", padding='same', input_shape=self.inputs_shape),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(2, 3, activation="relu", padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, activation="relu"),
        # tf.keras.layers.Dense(15, activation="relu"),
        tf.keras.layers.Dense(self.n_outputs, activation="softmax")

    ])
    self.model_train = model



    if self.train:
        self.logger.info("Setting up model from scratch.")
        self.model = self.model_train
    else:
        self.logger.info("Loading model from saved state.")
        try:
            self.model = tf.keras.models.load_model("saved_model")
        except:
            self.logger.debug("model cant be loaded from save place")


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    #print(game_state["field"])
    #print(game_state["self"])
    view_range = 3
    agents_position = game_state["self"][3]
    field = game_state["field"]
    #print(field)
    #print(agents_position)
    agents_view = np.ones((2*view_range+1,2*view_range+1))*-1
    agents_position_borders = np.broadcast_to(agents_position,(2,2))+np.array([[-1,-1],[+1,+1]])*view_range
    border_information_1 = np.where(agents_position_borders<0,-agents_position_borders,0)
    border_information_2 = np.where(agents_position_borders>16,agents_position_borders-16,0)
    border_information = border_information_2+border_information_1
    agents_position_borders = np.where(agents_position_borders < 0,0,agents_position_borders)
    agents_position_borders = np.where(agents_position_borders > 16,16,agents_position_borders)
    agents_real_view = field[agents_position_borders[0,0]:agents_position_borders[1,0]+1,agents_position_borders[0,1]:agents_position_borders[1,1]+1]

    x_spacing = (0+border_information[0,0],agents_view.shape[0]-border_information[1,0])
    y_spacing = (0+border_information[0,1],agents_view.shape[1]-border_information[1,1])

    agents_view[x_spacing[0]:x_spacing[1],y_spacing[0]:y_spacing[1]] = agents_real_view
    #print(agents_view)

    epsilon = max(1 - game_state["round"] / 500, 0.01)
    inputs = state_to_features(game_state)
    # todo Exploration vs exploitation

    self.logger.debug("Querying model for action.")


    if self.train:
        if np.random.rand() < epsilon:
            decision =  np.random.randint(5)
        else:
            Q_values = self.model.predict(inputs[np.newaxis])
            decision =  np.argmax(Q_values[0])
    else:
        decision = np.argmax(self.model.predict(inputs[np.newaxis]))
    #print(decision)
    #print(ACTIONS[decision])
    self.logger.info(f"Action {ACTIONS[decision]} at step {game_state['step']}")
    return ACTIONS[decision]


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
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    field = game_state["field"]
    agents_position = game_state["self"][3]
    inputs = np.zeros(field.shape + (2,))
    inputs[:, :, 0] = field
    inputs[:, :, 1][agents_position[0], agents_position[1]] = 1
    return inputs
