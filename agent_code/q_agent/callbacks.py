from collections import deque
import itertools
import os
import pickle
import random

import numpy as np
from sklearn import neighbors

from numpy import load


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
N_STATES = 825
M_ACTIONS = len(ACTIONS)
epsilon = 0.3 # exploration vs exproitation
dic = {} # for mapping states to index


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
    if self.train or not os.path.isfile("my-saved-model.npy"):
        self.logger.info("Setting up model from scratch.")
        q_table = np.zeros((N_STATES, M_ACTIONS))
        #q_table = np.random.rand(N_STATES, M_ACTIONS)
        self.model = q_table
        build_state_to_index()
        print("length of dict: ", len(dic))

    else:
        self.logger.info("Loading model from saved state.")
        #with open("my-saved-model.pt", "rb") as file:
        #    self.model = pickle.load(file)
        self.model = np.load("my-saved-model.npy")
        print(len(np.unique(self.model, axis = 0)))


def get_arrangements(array):
    return [ np.roll(array, -i) for i in range(0,4) ]

def get_variations(array, r=4):
    return [ item for item in list(itertools.product(array, repeat=r)) ]

# compute state to index for the Q-table
def build_state_to_index(arr1 = [-1,0,1,2], arr2 = [0,1,2]):
    perm = get_variations(arr1)
    #print(len(perm))
    for p in perm:
        arrangements = get_arrangements(p)
        for a in arrangements:
            completed_a = [np.append(a, j) for j in arr2] # add mode
            for c_a in completed_a:
                if tuple(c_a) in dic.keys():
                    #print("already there: ")
                    continue
                #print("add: ")
                i = len(dic)
                dic.update({tuple(c_a) : i})

# return corresponding index
def get_state_index(state):
    temp = get_arrangements(state[1:5])
    arrangements = [ np.append(t, state[-1]) for t in temp ]
    for a in arrangements:
        if tuple(a) in dic.keys():
            return dic[tuple(a)]
    print("Update dictionary")
    i = len(dic)
    dic.update({tuple(arrangements[0]) : i})
    return i


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    if (self.train and random.uniform(0, 1) < epsilon ) or game_state is None: # random.random()
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1]) # Explore action space

    self.logger.debug("Querying model for action.")
    
    state = state_to_features(game_state)
    #print("index",get_state_index(state))
    action = np.argmax(self.model[get_state_index(state)]) # Exploit learned values
    #print("values",self.model[get_state_index(state)])
    #print("action",action)
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

    # distance from position to objects
    def dist(pos, objects):
        return np.sqrt( np.power(np.subtract(objects, pos).transpose()[0], 2) + np.power(np.subtract(objects, pos).transpose()[1], 2) )
    
    
    # get region around the position
    def get_region(pos, n):
        vertical = np.array([ [x, pos[1]] for x in range(pos[0]-n, pos[0]+(n+1)) ])
        horizontal = np.array([ [pos[0], y] for y in range(pos[1]-n, pos[1]+(n+1)) ])
        return np.concatenate((vertical, horizontal), axis = 0)

    # check if there are collectable coins near position
    def collectable_coin(pos, n=4):
        if len(game_state['coins']) > 0: 
            return  game_state['coins'][np.argmin(dist(pos, game_state['coins']))] in get_region(pos, n) 
        else:
            return False

    # check if dangerous position
    def danger(pos, n=4):
        area = get_region(pos, n)
        bombs = game_state['bombs']
        if len(bombs) > 0:
            for bomb in bombs:
                for field in area:
                    if (np.array(bomb[0]) == field).all():
                        return True
        return False  


    # construct several channels of equal shape
    channels = []
    
    my_position = game_state['self'][3] # (x,y) coordinate of current position
    
    if True: # TODO value of current position
        my_position_value = 1
    
    channels.append(my_position_value)

    sub = [(1,0), (-1,0), (0,1), (0,-1)]
    neighbors = [ np.subtract(my_position, i) for i in sub ]
    neighbors_values = [ game_state['field'][neighbour[0]][neighbour[1]] for neighbour in neighbors ] # its entries are 1 for crates, −1 for stone walls and 0 for free tiles

    # calculate value of neighbours
    for j, i in enumerate(neighbors_values):
        if ((i == 0 and collectable_coin(neighbors[j])) or (i == 1 and not collectable_coin(neighbors[j]))) and not danger(neighbors[j]):
            channels.append(2) # depending on game mode drop bomb or go in this direction
        
        elif i != -1 and danger(neighbors[j]): 
            channels.append(1) # danger
        
        elif len(game_state['bombs']) > 0 and i == 0: 
            if (neighbors[j] == game_state['bombs'][0]).any():
                channels.append(-1) # bomb
            else: 
                channels.append(0) # free tiles
        
        elif len(game_state['others']) > 0 and i == 0: 
            if (neighbors[j] == game_state['others'][0][3]).any(): 
                channels.append(-1) # opponent
            else:
                channels.append(0) # free tiles
        
        elif i == 0:
            channels.append(0) # free tiles
        
        else:
            channels.append(-1) # crate or wall
    
    # calculate value of game mode
    mode = 0 # destroy crates
    if len(game_state['coins']) > 0 and np.min(dist(my_position, game_state['coins'])) < 5: # collectable coin near the agent
        mode = 2 # collect coins
    elif len(game_state['others']) > 0 and np.min(dist(my_position, game_state['others'])) < 4: # opponent near the agent
        mode = 1 # kill opponents

    channels.append(mode)
    
    # concatenate them as a feature tensor (they must have the same shape)
    stacked_channels = np.stack(channels)
    
    # and return them as a vector
    return stacked_channels.reshape(-1)

