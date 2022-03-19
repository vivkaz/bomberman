from collections import deque
from curses import A_CHARTEXT
import itertools
import os
import pickle
import random

import numpy as np
from sklearn import neighbors


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
N_STATES = 3072 # 4^4*3*4 = 3072 states, reduced by exploiting rotation
M_ACTIONS = len(ACTIONS)
#STEPS = 100
MAX_DIST_NEIGHBOUR = np.sqrt(8) # max distance from neighbour
MAX_DIST_FROM_ME = np.sqrt(32) # max distance from agent
epsilon = 0.4 # exploration vs exproitation?
dic = {} # for mapping states to indices

NEIGHBOURING_FIELDS = [-1,0,1,2] # the possible values for neighbouring fields
GAME_MODE = [0,1,2] # the possible values for game mode
CURRENT_FIELD = [0,1,2,3] # the possible values for current filed


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

    self.logger.info("Building dictionary for mapping states to indices.")

    build_state_to_index() 
    print(f"size of dictionary: {len(dic)}")

    if self.train or not os.path.isfile("my-saved-model.npy"): # how could we train the model again?
        self.logger.info("Setting up model from scratch.")
        q_table = np.zeros((N_STATES, M_ACTIONS)) # init Q-table
        #q_table = np.random.rand(N_STATES, M_ACTIONS) # find good initialization?
        self.model = q_table
        
    else:
        self.logger.info("Loading model from saved state.")
        self.model = np.load("my-saved-model.npy")

        print(f"# nonzero rows { sum(np.apply_along_axis(np.any, axis=1, arr=self.model)) }")

        #with open("my-saved-model.pt", "rb") as file:
        #    self.model = pickle.load(file)

# rotate clockwise
def get_arrangements(array):
    return np.unique([ np.roll(array, -i) for i in range(0,4) ], axis=0)

# get all possible combinations
def get_variations(array, r=4):
    return np.unique([ item for item in list(itertools.product(array, repeat=r)) ], axis=0)

# compute state to index for the Q-table
def build_state_to_index(arr1 = NEIGHBOURING_FIELDS, arr2 = GAME_MODE, arr3 = CURRENT_FIELD):
    perm = get_variations(arr1)
    for p in perm:
        arrangements = get_arrangements(p)
        for a in arrangements:
            #a_with_mode = [ np.append(a, j) for j in arr2 ] # add game mode
            comb = np.array(np.meshgrid(arr2, arr3)).T.reshape(-1,2) # combinations of arr2 and arr3
            a_completed = [ np.append(a, j) for j in comb ]
            for a_c in a_completed:
                if tuple(a_c) in dic.keys():
                    continue # already in dictionary
                i = len(dic)
                dic.update({tuple(a_c) : i})

# give corresponding index
def get_state_index(state):
    temp = get_arrangements(state[1:5]) # rotations of neighbouring fields
    mode_and_field = [ state[-1], state[0] ]
    arrangements = [ np.append(t, mode_and_field) for t in temp]
    for rotation, a in enumerate(arrangements):
        if tuple(a) in dic.keys():
            return dic[tuple(a)], rotation

    assert True == 0, "Dictionary should not be updated!"
    """i = len(dic)
    dic.update({tuple(arrangements[0]) : i})
    return i, 0"""


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    if (self.train and random.uniform(0, 1) < epsilon ) or game_state is None: # it was random.random()?
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1]) # Explore action space softmax ro ?

    self.logger.debug("Querying model for action.")
    
    state = state_to_features(game_state)
    index, rotation = get_state_index(state)
    action = np.argmax(self.model[index]) # Exploit learned values
    if action < 4 and rotation != 0: # move and rotated state
        action = (action + rotation) % 4 # compute rotated move

    # log and print actions while playing 
    #if not self.train:
    #    self.logger.info(f"Action {ACTIONS[action]} at step {game_state['step']}")
    #    print(f" STEP : {game_state['step']}, ACTION : {ACTIONS[action]}")
        
    return ACTIONS[action]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e. a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # distance from position to objects
    def dist(pos, objects):
        return np.sqrt( np.power(np.subtract(objects, pos).transpose()[0], 2) + np.power(np.subtract(objects, pos).transpose()[1], 2) )
    
    # get vertical and horizontal region from the position
    def get_vh_region(pos, n):
        horizontal = np.array([ [x, pos[1]] for x in range(pos[0]-n, pos[0]+(n+1)) ])
        vertical = np.array([ [pos[0], y] for y in range(pos[1]-n, pos[1]+(n+1)) ])
        return np.concatenate((vertical, horizontal), axis = 0)

    # get forward region from position
    def get_region(my_pos, pos):
        if pos[0] == my_pos[0] and pos[1] > my_pos[1]: # up
            r1 = np.array([ [pos[0]-1, y] for y in range(pos[1], pos[1]+3) ])
            r2 = np.array([ [pos[0], y] for y in range(pos[1], pos[1]+3) ])
            r3 = np.array([ [pos[0]+1, y] for y in range(pos[1], pos[1]+3) ])
            r12 = np.concatenate((r1, r2), axis = 0)
            return np.append(r12,r3).reshape(-1,2)
        elif pos[0] == my_pos[0] and pos[1] < my_pos[1]: # down
            r1 = np.array([ [pos[0]-1, y] for y in range(pos[1]-3, pos[1]) ])
            r2 = np.array([ [pos[0], y] for y in range(pos[1]-3, pos[1]) ])
            r3 = np.array([ [pos[0]+1, y] for y in range(pos[1]-3, pos[1]) ])
            r12 = np.concatenate((r1, r2), axis = 0)
            return np.append(r12,r3).reshape(-1,2)
        elif pos[1] == my_pos[1] and pos[0] > my_pos[0]: # right
            r1 = np.array([ [x, pos[1]+1] for x in range(pos[0], pos[0]+3) ])
            r2 = np.array([ [x, pos[1]] for x in range(pos[0], pos[0]+3) ])
            r3 = np.array([ [x, pos[1]-1] for x in range(pos[0], pos[0]+3) ])
            r12 = np.concatenate((r1, r2), axis = 0)
            return np.append(r12,r3).reshape(-1,2)
        else: # left
            r1 = np.array([ [x, pos[1]+1] for x in range(pos[0]-3, pos[0]) ])
            r2 = np.array([ [x, pos[1]] for x in range(pos[0]-3, pos[0]) ])
            r3 = np.array([ [x, pos[1]-1] for x in range(pos[0]-3, pos[0]) ])
            r12 = np.concatenate((r1, r2), axis = 0)
            return np.append(r12,r3).reshape(-1,2)


    # check if there are collectable coins near position
    def collectable_coin(pos, my_pos=(0,0), me=True):
        if len(game_state['coins']) > 0:
            if me == True: # check if coin in radius
                return np.min(dist(pos, game_state['coins'])) <= MAX_DIST_FROM_ME
            return game_state['coins'][np.argmin(dist(pos, game_state['coins']))] in get_region(my_pos, pos) # check if coin in region
        return False

        """area = get_region(my_pos, pos)
        coins = game_state['coins']
        if len(coins) > 0:
            for coin in coins:
                for field in area:
                    if coin == field:
                        return True
        else:
            return False"""

    # check if there are attackable opponents in radius of position
    def attackable_opponent(pos, r):
        if len(game_state['others']) > 0:
            coord_opponents = [other[3] for other in game_state['others']]
            return np.min(dist(pos, coord_opponents)) <= r
        return False

    def get_neighbours(pos): # check if in game border ? 
        sub = [(1,0), (0,1), (-1,0), (0,-1)] # left, down, right, up
        neighbours = [ np.subtract(pos, i) for i in sub ]
        return neighbours

    def get_neighbours_values(neighbours):
        neighbours_values = [ game_state['field'][neighbour[0]][neighbour[1]] for neighbour in neighbours ] # its entries are 1 for crates, −1 for stone walls and 0 for free tiles
        for j, value in enumerate(neighbours_values):
            if value == 0 and len(game_state['bombs']) > 0: 
                if (neighbours[j] == game_state['bombs'][0]).any():
                    neighbours_values[j] = -1 # bomb
            elif value == 0 and len(game_state['others']) > 0 :
                if (neighbours[j] == [other[3] for other in game_state['others']]).any():
                    neighbours_values[j] = -1 # opponent

        return neighbours_values

    # check if dangerous position
    def danger(pos, safe_danger_in = 3, n=3): # safe_danger_in?
        if len(game_state['bombs']) > 0:
            area = get_vh_region(pos, n)
            bombs = game_state['bombs']
            for bomb in bombs:
                for field in area:
                    if (np.array(bomb[0]) == field).all() and bomb[1] <= safe_danger_in:
                        return True
        return False

    def safe_death(pos):
        if danger(pos, 0): # safe death if do not move
            dangers = []
            neighbours = get_neighbours(pos)
            values = get_neighbours_values(neighbours)
            for i, value in enumerate(values):
                if danger(neighbours[i], 0): # safe death if move there
                    dangers.append(True)
                    continue
                if value != 0: # not free tile
                    dangers.append(True)
            return len(dangers) == 4 # could not move
        return False

    def blocked_opponent(opponent_pos, neighbours):
        if any([(opponent_pos[0] == neighbour[0] and opponent_pos[1] == neighbour[1]) for neighbour in neighbours]): # opponent is next to agent
            return safe_death(opponent_pos)
        return False

    def count_crates(pos):
        count = 0
        area = get_vh_region(pos, 3)
        for a in area:
            if game_state['field'][a[0]][a[1]] == 1: # crate
                count += 1
        return count

    def count_opponents(pos):
        count = 0
        area = get_vh_region(pos)
        others = game_state['others'][0][3]
        for opponent in others:
            for a in area:
                if a == opponent:
                    count += 1
        return count

    def coin_behind_crate(my_pos, crate_pos): # ?
        if len(game_state['coins']) > 0:
            coins = game_state['coins']
            if crate_pos[0] == my_pos[0] and crate_pos[1] > my_pos[1]: # up
                for coin in coins:
                    if coin[0] == crate_pos[0] and coin[1] == crate_pos[1] + 1:
                        return True
            elif crate_pos[0] == my_pos[0] and crate_pos[1] < my_pos[1]: # down
                for coin in coins:
                    if coin[0] == crate_pos[0] and coin[1] == crate_pos[1] - 1:
                        return True
            elif crate_pos[1] == my_pos[1] and crate_pos[1] > my_pos[1]: # right
                for coin in coins:
                    if coin[1] == crate_pos[1] and coin[0] == crate_pos[0] + 1:
                        return True
            elif crate_pos[1] == my_pos[1] and crate_pos[1] < my_pos[1]: # left
                for coin in coins:
                    if coin[1] == crate_pos[1] and coin[0] == crate_pos[0] - 1:
                        return True
        return False


    # construct several channels of equal shape
    channels = []
    
    my_position = game_state['self'][3] # (x,y) coordinates of current position
    neighbours = get_neighbours(my_position)
    neighbours_values = get_neighbours_values(neighbours)

    my_position_value = 0

    # TODO value of current position?
    # wait or bomb would lead to safe death
    if danger(my_position, 0):
        if not safe_death(my_position):
            my_position_value = 0 # move
        else: # safe death
            my_position_value = 1 # bomb

    # tricky: if neighbour is blocked opponent in danger, wait until agent is not in safe death       
    elif any(True == blocked_opponent(opponent, neighbours) for opponent in game_state['others']):
        my_position_value = 3 # kill blocked opponent 3 or 2=^wait ?

    # tricky: if in vh_region there are crate and coin in line, drop bomb 3 ?
    elif True == 1: 
        area = get_vh_region(my_position, 2)
        for a in area:
            if a[0] < 17 and a[1] < 17: # game board
                if game_state['field'][a[0]][a[1]] == 1:
                    if coin_behind_crate(my_position, a):
                        my_position_value = 3 # set a trap ? yourself be careful in such situations ? 

    # if there are many crates/opponents in agent's region, drop bomb
    elif (count_crates(my_position) + count_opponents(my_position)) > 5 and game_state['self'][2]==True: # ?
        my_position_value = 2 # drop bomb, destroy crates/ kill opponents

    elif (count_crates(my_position) + count_opponents(my_position)) > 3 and game_state['self'][2]==True: # ?
        my_position_value = 1 # drop bomb, destroy crates/ kill opponents

    # should wait
    elif all(True == danger(neighbour, 0) for neighbour in neighbours): #or game_state['self'][2]==False or count_crates(my_position) == 0: ?
        my_position_value = 2 # wait

    #else: #game_state['self'][2]==False: 
    #    my_position_value = 0 # move

    #elif ...:
    #    my_position_value = 1
        
    
    channels.append(my_position_value)


    # calculate value of neighbours
    for j, i in enumerate(neighbours_values):
        if  i != -1 and danger(neighbours[j]): 
            channels.append(1) # danger

        elif i == 0:
            if collectable_coin(neighbours[j], my_position, False): # coin
                channels.append(2) # depending on game mode go in this direction
            else: 
                channels.append(0) # free tile

        elif i == 1: # crate
            channels.append(2) # depending on game mode drop bomb
               
        else:
            channels.append(-1) # wall, opponent
    

    # calculate value of game mode
    mode = 0 # destroy crates
    if collectable_coin(my_position): # collectable coin near the agent
        mode = 2 # collect coins
    elif attackable_opponent(my_position, MAX_DIST_FROM_ME): # opponent near the agent
        mode = 1 # kill opponents

    channels.append(mode)
    
    
    # concatenate them as a feature tensor (they must have the same shape)
    stacked_channels = np.stack(channels)
    
    # and return them as a vector
    return stacked_channels.reshape(-1)

