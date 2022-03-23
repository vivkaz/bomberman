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
    #load_model = "agent/recent_best_coin_collector"
    #load_model = "saved_model_TASK_2-1"
    #load_model = "saved_model"
    #load_model = "saved_model_double_dqn_coin"

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


    #initialize an array for the current bombs, which are exploded
    self.exploded_bombs_normal = []#special case in train.py for old_state
    self.exploded_bombs_next = []#special case in train.py for new_state

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

    #print("callbacks - act")
    inputs = state_to_features(self,game_state)
    #print(f"inputs at setp {game_state['step']} : \n {inputs}")
    #print("inputs",inputs)
    # todo Exploration vs exploitation

    self.logger.debug("Querying model for action.")
    time_1 = time.time()

    if self.train:
        if np.random.rand() < epsilon:
            decision =  np.random.randint(self.Hyperparameter["n_outputs"])
        else:
            Q_values = self.model.predict(inputs[np.newaxis])[0]
            decision = np.argmax(Q_values)

        end_time = time.time()

        print("act_time : ", np.round(end_time - start_time, 6),"s")
        return ACTIONS[decision]
    else:
        Q_values = self.model.predict(inputs[np.newaxis])[0]#Q_values hat shape [1,2,3,4]
    time_2 = time.time()


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


def state_to_features(self,game_state: dict,mode = "normal") -> np.array:
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

    t_0 = time.time()

    try:
        size = self.Hyperparameter["field_size"]
    except:
        size = 17
    #print("size = ",size)


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
        border_information_2 = np.where(agents_position_borders > size -1 , agents_position_borders - (size -1), 0)
        border_information = border_information_2 + border_information_1
        agents_position_borders = np.where(agents_position_borders < 0, 0, agents_position_borders)
        agents_position_borders = np.where(agents_position_borders > (size -1), size-1, agents_position_borders)
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
    if x_coin.size != 0:
        coin_field[x_coin, y_coin] = 1

    x_bomb = []
    y_bomb = []
    bomb_timer = []
    bomb_position = game_state["bombs"]
    for i in bomb_position:
        x_bomb.append(i[0][0])
        y_bomb.append(i[0][1])
        bomb_timer.append(i[1])
    x_bomb = np.array(x_bomb)
    y_bomb = np.array(y_bomb)
    bomb_timer = np.array(bomb_timer)
    bomb_field = np.zeros(np.shape(field))
    bomb_timer_field = np.zeros(np.shape(field))
    if x_bomb != 0:
        bomb_field[x_bomb,y_bomb] = -2
        bomb_timer_field[x_bomb,y_bomb] = bomb_timer

    t_1 = time.time()


    def check_for_wall(position,field):
        #if (position >= 17).any() or (position < 0).any():
        #    print("out of field")
        #    return True
        if field[position[0],position[1]] == -1:
            #print("wall")
            return True
        else:
            return False

    def get_lethal_area(x_b, y_b):
        N = [np.array([x_b,y_b])]
        for direction in [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]:
            current_position = np.copy(np.array([x_b, y_b]))
            for i in range(3):
                current_position += direction
                if check_for_wall(current_position, field):
                    # print("break")
                    break
                else:
                    N.append(np.copy(current_position))
        return N


    #explosion_field = np.where(game_state["explosion_map"] != 0, -game_state["explosion_map"]-3,0)
    #print("explosion_field : ",explosion_field)
    # timer infromation auf die felder mit zukünftiger explosion erweitern
    #advanced_explosion_field = np.zeros(np.shape(field)) + explosion_field
    def get_advanced_explosion_field(bomb_buffer):
        advanced_explosion_field = np.zeros(np.shape(field))


        for count,(coord,timer) in enumerate(bomb_buffer):
            area = get_lethal_area(coord[0], coord[1])
            if timer == 4:
                value = 5
                bomb_buffer[count][1] = 5
            elif timer == 5:
                bomb_buffer[count][1] = 6
                value = 6
            else:
                continue
            for tile in area:
                advanced_explosion_field[tile[0],tile[1]] = value

        #print(f"bomb information : {game_state['bombs']} at step {game_state['step']}")



        for n in range(len(x_bomb)):
            x_b = x_bomb[n]
            y_b = y_bomb[n]
            timer = -bomb_timer[n]+4# t_new = -t_old + 4 mapping the timer on range 1-4 for 4 is setp before explosion
            #print("timer : ", timer)
            if timer == 4:
                bomb_buffer.append([np.array([x_b,y_b]),timer])

            area = get_lethal_area(x_b,y_b)
            #print(area)
            for coord in area:
                if advanced_explosion_field[coord[0],coord[1]] == 0 or advanced_explosion_field[coord[0],coord[1]] >= timer:
                    advanced_explosion_field[coord[0],coord[1]] = timer
        return advanced_explosion_field
    if mode == "normal":
        advanced_explosion_field = get_advanced_explosion_field(self.exploded_bombs_normal)
    elif mode == "next_state":
        advanced_explosion_field = get_advanced_explosion_field(self.exploded_bombs_next)

    t_2 = time.time()

    """
    advanced_explosion_field = np.zeros(np.shape(field))
    
    
        for count,(coord,timer) in enumerate(self.exploded_bombs):
            area = get_lethal_area(coord[0], coord[1])
            if timer == 0:
                value = -1
                self.exploded_bombs[count][1] = 1
            elif timer == 1:
                self.exploded_bombs[count][1] = 2
                value = -2
            else:
                continue
            for tile in area:
                advanced_explosion_field[tile[0],tile[1]] = value
    
        print(f"bomb information : {game_state['bombs']} at step {game_state['step']}")
    
    
    
        for n in range(len(x_bomb)):
            x_b = x_bomb[n]
            y_b = y_bomb[n]
            timer = bomb_timer[n]
            #print("timer : ", timer)
            if timer == 0:
                self.exploded_bombs.append([np.array([x_b,y_b]),timer])
    
            area = get_lethal_area(x_b,y_b)
            #print(area)
            for coord in area:
                if advanced_explosion_field[coord[0],coord[1]] == 0 or advanced_explosion_field[coord[0],coord[1]] >= timer:
                    advanced_explosion_field[coord[0],coord[1]] = timer
    """
                #print("add to explosion map at step : ", game_state["step"],"  : ",coord[0],coord[1],timer)
                #print("result : ", advanced_explosion_field[coord[0],coord[1]])

    #print(f"advanced_explosion_map at setp {game_state['step']} : {advanced_explosion_field}")


        #print("bomb",x_b,y_b,timer)


    """
        for direction in [np.array([0,1]),np.array([0,-1]),np.array([1,0]),np.array([-1,0])]:
            current_position = np.copy(np.array([x_b,y_b]))
            for i in range(3):
                current_position += direction
                #print("current_position", current_position)
                if check_for_wall(current_position,field):
                    #print("break")
                    break

                else:
                    if advanced_explosion_field[current_position[0],current_position[1]] == 0:#condition needed becuase a timer should not overwrite a current explosion or lower timer
                        advanced_explosion_field[current_position[0], current_position[1]] = timer
                        #print("timer_added")
        """
        #print(f"advanced_explosion map at step {game_state['step']} : {advanced_explosion_field}")


    #print("advanced_explosion_field",advanced_explosion_field)

    # input ["field_coin_map",view_range int, shape tuple, distance_information bool]

    def d(position, coins):
        return np.sqrt(np.power(coins[0] - position[0], 2) + np.power(coins[1] - position[1], 2))

    def field_coin_map(INPUT):
        view_range = INPUT[0]
        shape = INPUT[1]
        distance_information = INPUT[2]
        field_map = get_agents_view(field, view_range, agents_position, "field")
        coin_map = get_agents_view(coin_field, view_range, agents_position, "coin_field")

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
        #print(np.around(feature))
        #print(feature)
        return feature
    def one_field_map(INPUT):
        view_range = INPUT[0]
        shape = INPUT[1]#(v,v,1)
        field_map = get_agents_view(field, view_range, agents_position, "field")
        coin_map = get_agents_view(coin_field, view_range, agents_position, "coin_field")
        feature = np.zeros(shape)
        feature[:, :, 0] = field_map + coin_map
        return feature
    def fake_coin_field(INPUT):
        view_range = INPUT[0]
        shape = INPUT[1]
        field_map = get_agents_view(field, view_range, agents_position, "field")
        coin_map = get_agents_view(coin_field, view_range, agents_position, "coin_field")

        def check_index(index):
            if index > (size -1) :
                return size -1
            elif index < 0:
                return 0
            else:
                return index


        if np.sum(np.sum(coin_map,axis = 1),axis= 0 ) == 0 and x_coin.size != 0:
            agents_view_coord = []
            for i in range(-view_range,view_range+1,1):
                for j in range(-view_range,view_range+1,1):
                    if field[check_index(agents_position[0]+i),check_index(agents_position[1]+j)] == 0:
                        agents_view_coord.append([agents_position[0]+i,agents_position[1]+j])
            agents_view_coord = np.array(agents_view_coord)
            n = np.argmin(d(agents_position,np.array([x_coin, y_coin])))
            nearest_coin = np.array([x_coin[n],y_coin[n]])
            nearest_field = agents_view_coord[np.argmin(d(nearest_coin,agents_view_coord.transpose()))]
            #print("nearest_coin : ", nearest_coin)
            #print("nearest_field : ", nearest_field)
            agnets_coord_origin = agents_position - np.array([2,2])
            coin_map[nearest_field[0]-agnets_coord_origin[0],nearest_field[1]-agnets_coord_origin[1]] = 1

        feature = np.zeros(shape)
        feature[:, :, 0] = field_map
        feature[:, :, 1] = coin_map

        return feature


    def fake_coin_field_bombs(INPUT):
        view_range = INPUT[0]
        shape = INPUT[1]
        field_map = get_agents_view(field, view_range, agents_position, "field")
        coin_map = get_agents_view(coin_field, view_range, agents_position, "coin_field")
        bomb_map = get_agents_view(bomb_field,view_range,agents_position,"coin_field")#"coin_field" just sets tiles outside the field to zeros instead of -1 like for the option "field"
        advanced_explosion_map = get_agents_view(advanced_explosion_field,view_range,agents_position,"coin_field")

        def check_index(index):
            if index > size -1 :
                return size -1
            elif index < 0:
                return 0
            else:
                return index


        if np.sum(np.sum(coin_map,axis = 1),axis= 0 ) == 0 and x_coin.size != 0:
            agents_view_coord = []
            for i in range(-view_range,view_range+1,1):
                for j in range(-view_range,view_range+1,1):
                    if field[check_index(agents_position[0]+i),check_index(agents_position[1]+j)] == 0:
                        agents_view_coord.append([agents_position[0]+i,agents_position[1]+j])
            agents_view_coord = np.array(agents_view_coord)
            n = np.argmin(d(agents_position,np.array([x_coin, y_coin])))
            nearest_coin = np.array([x_coin[n],y_coin[n]])
            nearest_field = agents_view_coord[np.argmin(d(nearest_coin,agents_view_coord.transpose()))]
            #print("nearest_coin : ", nearest_coin)
            #print("nearest_field : ", nearest_field)
            agnets_coord_origin = agents_position - np.array([2,2])
            coin_map[nearest_field[0]-agnets_coord_origin[0],nearest_field[1]-agnets_coord_origin[1]] = 1


        #print(advanced_explosion_map)
        feature = np.zeros(shape)
        feature[:, :, 0] = (field_map + bomb_map)
        feature[:, :, 1] = coin_map
        feature[:,:,2] = advanced_explosion_map

        #print(advanced_explosion_map)
        return feature



    feature_functions = {"field_coin_map": field_coin_map,
                         "one_field_map" : one_field_map,
                         "fake_coin_field" : fake_coin_field,
                         "fake_coin_field_bombs": fake_coin_field_bombs}
    t_3 = time.time()

    inputs = feature_functions[self.Hyperparameter["feature_setup"]["feature_function"]](self.Hyperparameter["feature_setup"]["INPUTS"])

    t_4 = time.time()
    #print(f"input at step {game_state['step']} : {inputs}")
    times = np.array([t_4,t_3,t_2,t_1])-np.array([t_3,t_2,t_1,t_0])
    print("state to feature time : ", np.round(times,9))
    return inputs
