import os
import pickle
import random
from smtpd import DebuggingServer
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import deque


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ALPHA = 0.01
OPTIMIZER= Adam(learning_rate=ALPHA)


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
    #self.n_outputs = 5
    #self.inputs_shape = (9, 9, 2)
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, 4, activation="elu", padding='same', input_shape=self.inputs_shape),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(4, 3, activation="elu", padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(self.n_outputs, activation="softmax")

    ])
    """
    #name of loaded model either new initialize or pretrained model
    load_model = "initialize_model"
    #load_model = "saved_model_without_coin_9x9"


    try:
        self.model = tf.keras.models.load_model(load_model)
    except:
        self.logger.debug("model cant be loaded from save place")
    self.n_outputs = self.model.get_config()['layers'][-1]["config"]["units"]

"""
    # Build networks
    self.q_network = self._build_compile_model()
    self.target_network = self._build_compile_model()
    self.alighn_target_model()


def build_compile_model(self):
    model = Sequential()
    model.add(Embedding(200, 10, input_length=1))
    model.add(Reshape((10,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(200, activation='linear'))

    model.compile(loss='mse', oplimizer=OPTIMIZER)

    return model


def alighn_target_model(self):
    self.target_network.set_weights(self.q_network.get_wights)
"""

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # epsilon is a hyperparameter that introduces a random factor for the decisoin process for the first trained rounds.
    # This supports the agent to discover the enviroment
    epsilon = max(1 - game_state["round"] / 250, 0.1)#0.01)

    # inputs are the input data for the network, which calculates an action based on the inputs.
    # The function state_to_features transform the game states in to a 17x17xn matrix where the
    # information about position, walls, coins, ... are stored in the 3rd dimension
    inputs = state_to_features(game_state)
    
    # todo Exploration vs exploitation

    self.logger.debug("Querying model for action.")

    # actual decision process; the action with the highes Q-value is choosen
    if self.train:
        if np.random.rand() < epsilon:
            decision = np.random.randint(5)
        else:
            Q_values = self.model.predict(inputs[np.newaxis]) # q_values = self.q_network.predict(game_state)
            decision = np.argmax(Q_values[0])
    else:
        decision = np.argmax(self.model.predict(inputs[np.newaxis]))


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


    field = game_state["field"]
    agents_position = game_state["self"][3]
    coin_position = game_state["coins"]

    #implement the position of the coins
    x_coin = []
    y_coin = []
    for i in coin_position:
        x_coin.append(i[0])
        y_coin.append(i[1])
    x_coin = np.array(x_coin)
    y_coin = np.array(y_coin)

    #putting it all into the matrix inputs
    inputs = np.zeros(field.shape + (2,))
    inputs[:, :, 0] = field
    inputs[:, :, 1][agents_position[0], agents_position[1]] = 1
    inputs[:, :, 2][x_coin,y_coin] = 1

    return inputs
