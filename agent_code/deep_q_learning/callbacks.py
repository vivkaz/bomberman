import os
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
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
    #self.n_outputs = 5
    #self.inputs_shape = (9, 9, 2)

    #name of loaded model either new initialize or pretrained model
    if self.train:
        load_model = "initialize_model"
    else:
        load_model = "saved_model"
    #load_model = "saved_model"

    try:
        self.model = tf.keras.models.load_model(load_model)
        with open(f'{load_model}/Hyperparameter.pkl', 'rb') as f:
            self.Hyperparameter = pickle.load(f)
    except:
        self.logger.debug("model cant be loaded from save place")

    self.n_outputs = self.model.get_config()['layers'][-1]["config"]["units"]
    print(f"[info] loaded model to play/train : {load_model}")

    self.model_input_shape = self.model.get_config()["layers"][0]["config"]["batch_input_shape"]
    initial_predict = self.model.predict(np.zeros(self.model_input_shape[1:])[np.newaxis])
    del initial_predict

def act(self, game_state: dict) -> str:
    start_time = time.time()
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    #epsilon is a hyperparameter that introduces a random factor for the decisoin process for the first trained rounds. This supports the agent to discover the enviroment
    epsilon = max(1 - game_state["round"] / self.Hyperparameter["epsilon_scale"], 0.01)
    #epsilon = 0

    inputs = state_to_features(self,game_state)
    # todo Exploration vs exploitation

    self.logger.debug("Querying model for action.")
    time_1 = time.time()

    if self.train:
        if np.random.rand() < epsilon:
            decision =  np.random.randint(5)
        else:
            Q_values = self.model.predict(inputs[np.newaxis])[0]
            decision = np.argmax(Q_values)
        return ACTIONS[decision]
    else:
        Q_values = self.model.predict(inputs[np.newaxis])[0]#Q_values hat shape [1,2,3,4]
    time_2 = time.time()

    end_time = time.time()


    def invalid_move(action,game_state):
        if action < 4:

            old_position = np.array(game_state["self"][3])
            if action == 0:
                move = np.array([0,-1])
            elif action == 1:
                move = np.array([1,0])
            elif action == 2:
                move = np.array([0,1])
            elif action == 3:
                move = np.array([-1,0])
            new_position = old_position + move
            obsticle_coord = np.where((game_state["field"] == 1) | (game_state["field"] == -1))
            obsticles_on_field = np.array([obsticle_coord[0],obsticle_coord[1]]).transpose()

            if ((obsticles_on_field == np.broadcast_to(new_position,np.shape(obsticles_on_field))).all(axis = 1) == True).any():
                return True

            else:
                return False
        else:
            return False

    if invalid_move(np.argmax(Q_values),game_state):
        decision = np.where(Q_values == np.sort(Q_values)[-2])[0][0]
        print(f" invalid move occured at step : {game_state['step']} \n invalid move : {ACTIONS[np.argmax(Q_values)]} \n Q-values : {Q_values} \n"
              f" move replaced with : {ACTIONS[decision]}")
    else:
        decision = np.argmax(Q_values)



    self.logger.info(f"Action {ACTIONS[decision]} at step {game_state['step']}")

    print(
        f" STEP : {game_state['step']}, ACTION : {ACTIONS[decision]} with time : {np.round(end_time - start_time, 3)},{np.round(time_1 - start_time, 3)},{np.round(time_2 - start_time, 3)}")

    print(f"Q_Values : {self.model.predict(inputs[np.newaxis])}")


    return ACTIONS[decision]


def state_to_features(self,game_state: dict) -> np.array:
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
    def get_agents_view(map,view_range,agents_position,type):
        if type == "field":
            agents_view = np.ones((2 * view_range + 1, 2 * view_range + 1)) * -1
        elif type == "coin_field":
            agents_view = np.zeros((2 * view_range + 1, 2 * view_range + 1))
        agents_position_borders = np.broadcast_to(agents_position, (2, 2)) + np.array([[-1, -1], [+1, +1]]) * view_range
        border_information_1 = np.where(agents_position_borders < 0, -agents_position_borders, 0)
        border_information_2 = np.where(agents_position_borders > 16, agents_position_borders - 16, 0)
        border_information = border_information_2 + border_information_1
        agents_position_borders = np.where(agents_position_borders < 0, 0, agents_position_borders)
        agents_position_borders = np.where(agents_position_borders > 16, 16, agents_position_borders)
        agents_real_view = map[agents_position_borders[0, 0]:agents_position_borders[1, 0] + 1,
                           agents_position_borders[0, 1]:agents_position_borders[1, 1] + 1]

        x_spacing = (0 + border_information[0, 0], agents_view.shape[0] - border_information[1, 0])
        y_spacing = (0 + border_information[0, 1], agents_view.shape[1] - border_information[1, 1])

        agents_view[x_spacing[0]:x_spacing[1], y_spacing[0]:y_spacing[1]] = agents_real_view
        return agents_view


    agents_position = game_state["self"][3]
    field = game_state["field"]

    x_coin = []
    y_coin = []
    coin_position = game_state["coins"]
    for i in coin_position:
        x_coin.append(i[0])
        y_coin.append(i[1])
    x_coin = np.array(x_coin)
    y_coin = np.array(y_coin)
    coin_field = np.zeros(np.shape(field))
    coin_field[x_coin, y_coin] = 1


    # input ["field_coin_map",view_range int, shape tuple, distance_information bool]
    def field_coin_map(INPUT):
        view_range = INPUT[0]
        shape = INPUT[1]
        distance_information = INPUT[2]
        field_map = get_agents_view(field, view_range, agents_position, "field")
        coin_map = get_agents_view(coin_field, view_range, agents_position, "coin_field")
        def d(position, coins):
            return np.sqrt(np.power(coins[0] - position[0], 2) + np.power(coins[1] - position[1], 2))
        if distance_information == True:
            if x_coin.size != 0:
                d_coin_min = np.min(d(agents_position, np.array([x_coin, y_coin])))
                coin_map[int(np.shape(coin_map)[0] / 2), int(np.shape(coin_map)[1] / 2)] = d_coin_min / (
                            np.sqrt(2) * 15)
            else:
                d_coin_min = 0

        feature = np.zeros(shape)
        feature[:,:,0] = field_map
        feature[:,:,1] = coin_map
        return feature

    feature_functions = {"field_coin_map": field_coin_map
                         }


    inputs = feature_functions[self.Hyperparameter["feature_setup"]["feature_function"]](self.Hyperparameter["feature_setup"]["INPUTS"])

    return inputs
