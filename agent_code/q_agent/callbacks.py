from collections import deque
#from curses import A_CHARTEXT
import itertools
from lib2to3.pytree import LeafPattern
import os
import pickle
import random
from re import T

import numpy as np
from sklearn import neighbors

NEIGHBOURING_FIELDS = [-1,0,1,2] # the possible values for neighbouring fields
GAME_MODE = [0,1,2,3] # the possible values for game mode
CURRENT_FIELD = [0,1,2] # the possible values for current filed

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
N_STATES = 70*len(GAME_MODE)*len(CURRENT_FIELD) # 4^4*3*4 = 3072 states, reduced to 1120 by exploiting rotation
M_ACTIONS = len(ACTIONS)
#STEPS = 100
MAX_DIST_NEIGHBOUR = 3 # max distance from neighbour
MAX_DIST_FROM_ME = 3 #np.sqrt(10) # max distance from agent
epsilon = 0.1 # exploration vs exproitation?
dic = {} # for mapping states to indices


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
    #print(f"size of dictionary: {len(dic)}") # should be 55, but it is 70 (we exploit rotation but not mirrot invariance)

    """if self.train or not os.path.isfile("my-saved-model.npy"): # how could we train the model again ? 
        self.logger.info("Setting up model from scratch.")
        q_table = np.zeros((N_STATES, M_ACTIONS)) # init Q-table

        # random samples from a uniform distribution over [0, 1)
        #q_table = np.random.rand(N_STATES, M_ACTIONS) # find good initialization ?
        self.model = q_table
        
    else:
        self.logger.info("Loading model from saved state.")
        self.model = np.load("my-saved-model.npy")

        print(f"nonzero rows: { sum(np.apply_along_axis(np.any, axis=1, arr=self.model)) }")

        #with open("my-saved-model.pt", "rb") as file:
        #    self.model = pickle.load(file)"""

    self.model = np.load("my-saved-model.npy")

# rotate clockwise
def get_arrangements(array):
    return [ np.roll(array, -i) for i in range(0,4) ]

# get all possible combinations
def get_variations(array, r=4):
    return [ item for item in list(itertools.product(array, repeat=r)) ]

# compute state to index for the Q-table
def build_state_to_index(arr1 = NEIGHBOURING_FIELDS, arr2 = GAME_MODE, arr3 = CURRENT_FIELD):
    i = 0
    perm = get_variations(arr1)
    comb = np.array(np.meshgrid(arr2, arr3)).T.reshape(-1,2) # combinations of arr2 and arr3
    step = len(comb)
    for p in perm:
        already_there = []
        arrangements = get_arrangements(p)
        for a in arrangements:
            if tuple(a) in dic.keys():
                already_there.append(True)
            else:
                already_there.append(False)
        if sum(already_there) == 0: # p not in dictionary
            value = [v for v in range(i, i+step)] # value indices for p
            i += step # update i
            dic.update({tuple(p) : value}) # add in dictionary

# give corresponding index and rotation from keys
def get_state_index(state):
    arrangements = get_arrangements(state[-4:]) # rotations of neighbouring fields
    mode_and_field = state[:2]
    value_index = 0
    comb = np.array(np.meshgrid(GAME_MODE, CURRENT_FIELD)).T.reshape(-1,2)
    for i, c in enumerate(comb):
        if (c == mode_and_field).all():
            value_index = i
            break
    for j, a in enumerate(arrangements):
        if tuple(a) in dic.keys():
            return dic[tuple(a)][value_index], j
    assert True == 0, state


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    #epsilon = np.random.choice( [max(1 - game_state["round"] / 500, 0.05), 0.1], p=[.7, .3]) 
    epsilon = max(1 - game_state["round"] / 1000, 0.1)

    if (self.train and random.random() < epsilon): # random.uniform() or random.rand() ?
        self.logger.debug("Choosing action purely at random.")
        #print("\nrandom")
        # 80%: walk in any direction. 10% wait. 10% bomb. (random action is selected with regards to the weight associated with each action)
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1]) # softmax; e-greedy or e-soft (select random actions uniformly with probability e or 1-e) instead ?

    self.logger.debug("Querying model for action.")
    state = state_to_features(game_state)
    index, rotation = get_state_index(state)    
    action = np.argmax(self.model[index]) # Exploit learned values
    
    print("\state, i, r, a",state, index, rotation, action)
    print("model", self.model[index])

    if action < 4 and rotation != 0: # move and rotated state
        action = (action + (4-rotation)) % 4 # compute rotated move
        #print("new action",action)

    print(ACTIONS[action], "\n") 

    # log and print actions while playing 
    if not self.train:
        self.logger.info(f"Action {ACTIONS[action]} at step {game_state['step']}")
        #print(f" STEP : {game_state['step']}, ACTION : {ACTIONS[action]}")
        #print ("q-table",self.model[index]) 

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
    #if game_state is None:
    #    return None

    def get_neighbours(pos):
        sub = [(1,0), (0,-1), (-1,0), (0,1)] # left, down, right, up
        neighbours = []
        for i in sub:
            neighbour = np.subtract(pos, i)
            if (0 <= neighbour[0] < 17) and (0 <= neighbour[1] < 17): # game borders
                neighbours.append(neighbour)
        return neighbours

    def get_neighbours_values(neighbours):
        neighbours_values = [ game_state['field'][neighbour[0]][neighbour[1]] for neighbour in neighbours ] # −1=walls,  0=free tiles, 1=crates
        for j, value in enumerate(neighbours_values):
            if value == 0 and len(game_state['bombs']) > 0: 
                if any([ (neighbours[j] == np.array(bomb[0])).all() for bomb in game_state['bombs']]):
                    neighbours_values[j] = 2 # bomb
            elif value == 0 and len(game_state['others']) > 0 :
                if any([(neighbours[j] == np.array(other[3])).all() and (other[3] != game_state['self'][3]) for other in game_state['others'] ]):
                    neighbours_values[j] = 1 # opponent
            elif value == 0 and game_state['explosion_map'][neighbours[j][0]][neighbours[j][1]] == 1:
                neighbours_values[j] = 2  # explosion
            print("j explosion", j, game_state['explosion_map'][neighbours[j][0]][neighbours[j][1]])
        return neighbours_values

    # distance from position to objects
    def dist(pos, objects):
        return np.sqrt( np.power(np.subtract(objects, pos).transpose()[0], 2) + np.power(np.subtract(objects, pos).transpose()[1], 2) )
    
    def check_for_wall(y, x, field=game_state['field']):
        if (0 <= x < 17) and (0 <= y < 17):
            return field[y][x] == -1 # wall
        return False

    def get_left(pos, n=3):
        left = []
        for y in range(pos[0]-1, pos[0]-(n+1), -1):
            if (0 <= y < 17):
                if check_for_wall(y, pos[1]):
                    break
                else:
                    left.append([y, pos[1]])
        return left

    def get_right(pos, n=3):
        right = []
        for y in range(pos[0]+1, pos[0]+(n+1)):
            if (0 <= y < 17):
                if check_for_wall(y, pos[1]):
                    break
                else:
                    right.append([y, pos[1]])
        return right

    def get_up(pos, n=3):
        up = []
        for x in range(pos[1]-1, pos[1]-(n+1), -1):
            if (0 <= x < 17):
                if check_for_wall(pos[0], x):
                    break
                else:
                    up.append([pos[0], x])
        return up

    def get_down(pos, n=3):
        down = []
        for x in range(pos[1]+1, pos[1]+(n+1)):
            if (0 <= x < 17):
                if check_for_wall(pos[0], x):
                    break
                else:
                    down.append([pos[0], x])
        return down

    # get vertical and horizontal region from the position, consider walls
    def get_vh_region(pos, n=3):
        left = get_left(pos, n)
        right = get_right(pos, n)
        up = get_up(pos, n)
        down = get_down(pos, n)
        
        for r in right:
            left.append(r)
        for u in up:
            left.append(u)
        for d in down:
            left.append(d)
        return left

    # get forward region from position; observe crates and walls ?
    def get_region(my_pos, pos, n=2):
        if pos[1] == my_pos[1] and pos[0] < my_pos[0]: # left
            r1 = np.array([ [x, pos[1]-1] for x in range(pos[0]-n, pos[0]+1) ])
            r2 = np.array([ [x, pos[1]] for x in range(pos[0]-n, pos[0]+1) ])
            r3 = np.array([ [x, pos[1]+1] for x in range(pos[0]-n, pos[0]+1) ])
            r12 = np.concatenate((r1, r2), axis = 0)
            return np.append(r12,r3).reshape(-1,2)
        elif pos[1] == my_pos[1] and pos[0] > my_pos[0]: # right
            r1 = np.array([ [x, pos[1]-1] for x in range(pos[0], pos[0]+n+1) ])
            r2 = np.array([ [x, pos[1]] for x in range(pos[0], pos[0]+n+1) ])
            r3 = np.array([ [x, pos[1]+1] for x in range(pos[0], pos[0]+n+1) ])
            r12 = np.concatenate((r1, r2), axis = 0)
            return np.append(r12,r3).reshape(-1,2)
        elif pos[0] == my_pos[0] and pos[1] > my_pos[1]: # down
            r1 = np.array([ [pos[0]-1, y] for y in range(pos[1], pos[1]+n+1) ])
            r2 = np.array([ [pos[0], y] for y in range(pos[1], pos[1]+n+1) ])
            r3 = np.array([ [pos[0]+1, y] for y in range(pos[1], pos[1]+n+1) ])
            r12 = np.concatenate((r1, r2), axis = 0)
            return np.append(r12,r3).reshape(-1,2)
        else: # up
            r1 = np.array([ [pos[0]-1, y] for y in range(pos[1]-n, pos[1]+1) ])
            r2 = np.array([ [pos[0], y] for y in range(pos[1]-n, pos[1]+1) ])
            r3 = np.array([ [pos[0]+1, y] for y in range(pos[1]-n, pos[1]+1) ])
            r12 = np.concatenate((r1, r2), axis = 0)
            return np.append(r12,r3).reshape(-1,2)

    """def get_square(pos):
        r1 = np.array([ [x, pos[1]-2] for x in range(pos[0]-1, pos[0]+1) ])
        r2 = np.array([ [x, pos[1]-1] for x in range(pos[0]-2, pos[0]+2) ])
        r3 = np.array([ [x, pos[1]] for x in range(pos[0]-2, pos[0]+2) ])
        r23 = np.concatenate((r2, r3), axis = 0)
        r4 = np.array([ [x, pos[1]+1] for x in range(pos[0]-2, pos[0]+2) ])
        r5 = np.array([ [x, pos[1]+2] for x in range(pos[0]-1, pos[0]+1) ])
        r15 = np.concatenate((r1, r5), axis = 0)
        r234 = np.append(r23,r4).reshape(-1,2)
        return np.append(r234,r15).reshape(-1,2)"""

    # check if there are collectable coins near position
    def collectable_coin(pos, neighbours=[], my_pos=(0,0), me=True):
        if len(game_state['coins']) > 0:
            coins = game_state['coins']
            if me == True: # check if coin in agent's radius
                return np.min(dist(pos, coins)) <= MAX_DIST_FROM_ME, -1
        
            neighbours_values = get_neighbours_values(neighbours)
            n_coins = 0  # number coins in region
            j = -1 # neighbour with most coins in region
            k = -1 # neighbour with min distance to a coin
            distance = 2.2 # ?
            for i, neighbour in enumerate(neighbours):
                if neighbours_values[i] == 0: # free tile
                    d = np.min(dist(neighbour, coins))
                    if d < distance:
                        distance, k = d, i
                    region = get_region(my_pos, neighbour,3)
                    temp = 0
                    for coin in coins:
                        temp += len(list(filter (lambda x : (x == np.array(coin)).all(), region)))
                    if temp > n_coins:
                        n_coins, j = temp, i
            
            if k != -1:
                return True, k
            else:
                return True, j
        return False, -1

    # check if there are attackable opponents in radius of position
    def attackable_opponent(pos, my_pos=(0,0), me=True):
        if len(game_state['others']) > 0:
            opponents = [ other[3] for other in game_state['others'] ]
            if me == True: # check if opponent in agent's radius
                return np.min(dist(pos, opponents)) <= MAX_DIST_FROM_ME
            
            if np.min(dist(pos, opponents)) <= MAX_DIST_NEIGHBOUR:
                opponent = game_state['others'][np.argmin(dist(pos, opponents))][3] # nearest opponent
                region = get_region(my_pos, pos)
                return (len(list(filter (lambda x : (x == np.array(opponent)).all(), region))) > 0) # check if in neighbour's region
        return False

    # nearest crate
    def destroyable_crate(neighbours):
        crate_coord = np.where(game_state["field"] == 1)
        crates = np.array([crate_coord[0], crate_coord[1]]).transpose()
        mask = np.where([game_state["field"][neighbour[0]][neighbour[1]] != -1 for neighbour in neighbours])
        if len(mask) > 0:
            temp = [neighbours[i] for i in mask[0]]
            d = np.array([ np.min(dist(neighbour, crates)) for neighbour in temp ])
            return mask[0][np.where(d == min(d))]

            #return mask[0][np.argmin([ np.min(dist(neighbour, crates)) for neighbour in temp ])]
        return -1

    def count_free_tiles(my_pos, pos):
        if game_state['field'][pos[0]][pos[1]] == 0:
            count = 0
            next_pos = (0,0)
            if pos[1] == my_pos[1] and pos[0] < my_pos[0]: # left
                next_pos = (pos[0]-1, pos[1])
            elif pos[1] == my_pos[1] and pos[0] > my_pos[0]: # right
                next_pos = (pos[0]+1, pos[1])
            elif pos[0] == my_pos[0] and pos[1] > my_pos[1]: # down
                next_pos = (pos[0], pos[1]+1)
            else: # up
                next_pos = (pos[0], pos[1]-1)
                
            area = get_neighbours(next_pos)
            area.append(np.array(next_pos))
            for a in area:
                if game_state['field'][a[0]][a[1]] == 0: # free
                    count += 1
            return count
        return -1

    def direction_to_escape_bomb(my_pos, bombs, neighbours, j):
        index = np.argmin( [ dist(my_pos, bomb[0]) for bomb in bombs ])
        bomb_area = np.array(get_vh_region(bombs[index][0]))
        escape_index = np.where([neighbour not in bomb_area for neighbour in neighbours])[0]
        print("escape i j",escape_index, j)
        if len(escape_index) > 0 and  escape_index[0] == j: 
            return True

        return np.argmax([ count_free_tiles(my_pos, neighbour) for neighbour in neighbours ]) == j

    # check if dangerous position
    def danger(pos, safe_danger_in=3, n=3): # timer=3 means that bomb was dropped in previous step
        if len(game_state['bombs']) > 0 and game_state['field'][pos[0]][pos[1]] != -1: # (for neighbours check if not a wall)
            bombs = game_state['bombs']
            area = get_vh_region(pos, n)
            area.append([pos[0], pos[1]])
            print("area", area)
            for a in area:
                print("in danger", [ ( (np.array(bomb[0]) == a).all() and bomb[1] <= safe_danger_in ) for bomb in bombs])
                if any([ ( (np.array(bomb[0]) == a).all() and bomb[1] <= safe_danger_in ) for bomb in bombs]):
                    return True
        return False

    def safe_death(pos):
        if danger(pos, 0): # safe death if do not move, bomb will explode in next step
            dangers = []
            neighbours = get_neighbours(pos)
            values = get_neighbours_values(neighbours)
            for i, value in enumerate(values):
                if value != 0: # not free tile
                    dangers.append(True)
                elif danger(neighbours[i], 0 ) or game_state['explosion_map'][neighbours[i][0]][neighbours[i][1]] == 1: # safe death if move there
                    dangers.append(True)
            return len(dangers) == 4 # could not move
        return False

    def blocked_opponent(opponent_pos, my_neighbours):
        if any([ (np.array(opponent_pos) == neighbour).all() for neighbour in my_neighbours ]): # opponent is neighbour of agent
            return safe_death(opponent_pos)
        return False

    # count crates in bomb region
    def count_crates(pos):
        count = 0
        area = get_vh_region(pos)
        for a in area:
            if game_state['field'][a[0]][a[1]] == 1: # crate
                count += 1
        return count

    # count opponents in bomb region
    def count_opponents(pos):
        if len(game_state['others']) > 0:
            count = 0
            area = get_vh_region(pos)
            others = game_state['others'][0][3]
            for opponent in others:
                count += (len(list(filter (lambda x : (x == np.array(opponent)).all(), area)))) 
        return count

    def coin_behind_crate(my_pos, crate_pos):
        if len(game_state['coins']) > 0:
            coins = game_state['coins']
            if crate_pos[1] == my_pos[1] and crate_pos[0] < my_pos[0]: # left
                for coin in coins:
                    if coin[1] == crate_pos[1] and coin[0] == crate_pos[0] - 1:
                        return True
            elif crate_pos[1] == my_pos[1] and crate_pos[0] > my_pos[0]: # right
                for coin in coins:
                    if coin[1] == crate_pos[1] and coin[0] == crate_pos[0] + 1:
                        return True
            elif crate_pos[0] == my_pos[0] and crate_pos[1] > my_pos[1]: # down
                for coin in coins:
                    if coin[0] == crate_pos[0] and coin[1] == crate_pos[1] + 1:
                        return True
            elif crate_pos[0] == my_pos[0] and crate_pos[1] < my_pos[1]: # up
                for coin in coins:
                    if coin[0] == crate_pos[0] and coin[1] == crate_pos[1] - 1:
                        return True
        return False



    """ CONSTRUCT FEATURES """
    channels = []
    
    my_position = game_state['self'][3] # (y, x) coordinates of current position
    neighbours = get_neighbours(my_position)
    neighbours_values = get_neighbours_values(neighbours)

    print("my_pos ", my_position)


    """ calculate value of game mode """
    mode = 0 # destroy crates
    if danger(my_position):
        mode = 2 # escape, especially when droped bomb
    elif attackable_opponent(my_position) or (safe_death(my_position) and game_state['self'][2]==True): # opponent near the agent or safe death
        mode = 1 # kill opponents
    elif collectable_coin(my_position)[0] == True: # collectable coin near the agent
        mode = 3 # collect coins

    channels.append(mode)


    """ calculate value of current position """
    my_position_value = 0 # move

    # wait or bomb would lead to safe death
    if danger(my_position):
        if safe_death(my_position) and game_state['self'][2]==True:
            my_position_value = 1 # bomb because nothing else to do 
     
    # tricky: if neighbour is blocked opponent in danger, wait until agent is not in safe death  
    elif any(True == blocked_opponent(opponent[3], neighbours) for opponent in game_state['others']):
        my_position_value = 2 # kill blocked opponent by waiting
        
    # there are crates/opponents in agent's region, drop bomb
    elif 1 in neighbours_values and game_state['self'][2]==True:
        my_position_value = 1 # drop bomb, destroy crates/ kill opponents

        #if (count_crates(my_position) + count_opponents(my_position)) >= 4 and game_state['self'][2]==True:
        #    my_position_value = 3 # drop bomb, destroy crates/ kill opponents
        #if (count_crates(my_position) + count_opponents(my_position)) >= 1 and game_state['self'][2]==True:
        #    my_position_value = 1 # drop bomb, destroy crates/ kill opponents

    elif all( ( danger(neighbour, 0)==True or game_state['explosion_map'][neighbour[0]][neighbour[1]]==1 or game_state['field'][neighbour[0]][neighbour[1]] != 0 ) for neighbour in neighbours):
        my_position_value = 2 # should wait

    elif all([ neighbours_values[j] != 0 for j in range(4)]):
        my_position_value = 2 # should wait

    # tricky: if in vh_region there are crate and coin in line, drop bomb
    else: 
        area = get_vh_region(my_position, 2)
        for a in area:
            if game_state['field'][a[0]][a[1]] == 1:
                if coin_behind_crate(my_position, a):
                    my_position_value = 1 # set a trap, drop a bomb

    #elif ...:
    #    my_position_value = ...

    channels.append(my_position_value)


    """ calculate values of neighbours """
    crate_dir_flag = False
    for j, value in enumerate(neighbours_values):
        if value == 2: # bomb or explosion
            channels.append(2) # danger

        elif value == 0:
            if danger(my_position):
                if direction_to_escape_bomb(my_position, game_state['bombs'], neighbours, j):
                    channels.append(1) # direction to escape
            
            elif danger(neighbours[j]) or game_state['explosion_map'][neighbours[j][0]][neighbours[j][1]]==1:
                channels.append(2) # danger
            
            elif collectable_coin(neighbours[j], neighbours, my_position, False)[1] == j or attackable_opponent(neighbours[j], my_position, False): # coin or opponent in region
                channels.append(1) # depending on game mode go in this direction or drop bomb
            
            elif ( j in destroyable_crate(neighbours) ) and not crate_dir_flag:
                crate_dir_flag = True
                channels.append(1) # depending on game mode go in this direction or drop bomb

            else:
                channels.append(0) # free

        elif value == 1 and mode != 2: # crate or opponent, not in danger
            channels.append(1)

        else:
            channels.append(-1) # wall
    
    assert len(channels) == 6, "Not all features in channels"

    print("features",channels)

    # concatenate them as a feature tensor (they must have the same shape)
    stacked_channels = np.stack(channels)
    
    # and return them as a vector
    return stacked_channels.reshape(-1)


