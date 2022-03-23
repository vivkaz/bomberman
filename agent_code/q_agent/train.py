from collections import namedtuple, deque
from datetime import datetime
from pathlib import PosixPath
import random
#from turtle import position
from matplotlib.pyplot import new_figure_manager
import numpy as np

import pickle
from typing import List

import events as e
from .callbacks import get_state_index, state_to_features, get_arrangements, ACTIONS, dic, epsilon, GAME_MODE, CURRENT_FIELD

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 4  # keep only 4 last transitions
BUFFER_HISTORY_SIZE = 6 # keep only 6 last states
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability 
# determines to what extent newly acquired information overrides old information
ALPHA = 0.1 # in fully deterministic environments 1 is optimal; when problem is stochastic, often const learning rate such as 0.1
# determines the importance of future rewards
GAMMA = 0.9

# Events
#PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # save rewards
    self.rewards = []

    # buffer with last 4 positions for checking if agent runs in loop
    self.buffer_states = deque(maxlen=BUFFER_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    def update_buffer_states():
        if new_game_state is not None:
            self.buffer_states.append(new_game_state['self'][3]) # agents coordinates on field

    def in_danger(area, bombs):
        for bomb in bombs:
            for a in area:
                #if (bomb == a).all():
                if (np.array(bomb[0]) == a).all():
                    return True
        return False

    #def bomb_dropped(): # covered in environment.py
    #    return old_game_state['self'][2] == True and new_game_state['self'][2] == False

    def bomb_avoided():
        def check_for_wall(y, x, field=old_game_state['field']):
            if (0 <= x < 17) and (0 <= y < 17):
                return field[y][x] == -1 # wall
            return False

        def get_left(pos, n=3):
            left = []
            for y in range(pos[0]-n, pos[0]):
                if (0 <= y < 17):
                    if check_for_wall(y, pos[1]):
                        left = []
                        continue
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
            for x in range(pos[1]-n, pos[1]):
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
                        down = []
                        continue
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
                down.append(u)
            for d in down:
                left.append(d)
            return left


        if len(old_game_state['bombs']) > 0:
            bombs = old_game_state['bombs']
            old_risky_area = get_vh_region(old_game_state['self'][3])
            new_risky_area = get_vh_region(new_game_state['self'][3])

            old_danger = in_danger(old_risky_area, bombs)
            new_danger = in_danger(new_risky_area, bombs)
            return old_danger and not new_danger

        """bombs = np.empty((2, len(old_game_state['bombs'])))
        if bombs.size != 0:
            for count, bomb in enumerate(old_game_state['bombs']):
                bombs[0, count] = bomb[0][0]
                bombs[1, count] = bomb[0][1]
            bombs = bombs.transpose()
            old_danger = in_danger(old_risky_area, bombs)
            new_danger = in_danger(new_risky_area, bombs)
            return old_danger and not new_danger"""
    
    def get_collectable_coins():
        coins = np.empty((2, len(old_game_state['coins'])))
        if coins.size != 0:
            for count, coin in enumerate(old_game_state['coins']):
                coins[0, count] = coin[0]
                coins[1, count] = coin[1]
        return coins, coins.size != 0

    def coin_distance_reduced():
        old_position = old_game_state['self'][3]
        new_position = new_game_state['self'][3]
        coins, flag = get_collectable_coins()

        if flag: # coins not empty
            def dist(position, coins):
                return np.sqrt( np.power(coins[0] - position[0], 2) + np.power(coins[1] - position[1], 2) )
            
            old_dist = dist(old_position, coins)
            new_dist = dist(new_position, coins)
            return flag, np.min(new_dist) < np.min(old_dist), np.min(new_dist) == np.min(old_dist)
        else: 
            return flag, False, False
            
    def run_in_loop():
        update_buffer_states()
        if len(self.buffer_states) == BUFFER_HISTORY_SIZE:
            #return (self.buffer_states[0] == self.buffer_states[2]).all() and (self.buffer_states[1] == self.buffer_states[3]).all() and (self.buffer_states[0] != self.buffer_states[1]).any()
            buffer = np.array(self.buffer_states)
            return len(np.unique(buffer[[0, 2, 4]], axis=0)) == 1 and len(
                        np.unique(buffer[[1, 3, 5]], axis=0)) == 1 and (
                            np.unique(buffer[[0, 2, 4]], axis=0) != np.unique(buffer[[1, 3, 5]], axis = 0)).any()        
        else:
            return False


    """Add your own events to hand out rewards"""
    if old_game_state is not None:
        if bomb_avoided():
            events.append(e.BOMB_AVOIDED)

        #if bomb_dropped(): #covered in environment.py
        #    events.append(e.BOMB_DROPPED)

        collectable_coins, reduced_dist, same_distance = coin_distance_reduced()
        if collectable_coins: 
            if reduced_dist:
                events.append(e.COIN_DISTANCE_REDUCED)
            elif not same_distance: 
                events.append(e.COIN_DISTANCE_INCREASED)

    if run_in_loop():
        events.append(e.RUN_IN_LOOP)

    #print("events", events)
     
    
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    #print((f"Step {new_game_state['step']} : {events}"))

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    # Recalculate Q-values
    state, action, next_state, reward = self.transitions[-1]

    # check if state is None
    if next_state is not None:
   
        index, rotation = get_state_index(state)

        action = ACTIONS.index(action)

        if action < 4 and rotation != 0:
            action = (action + rotation) % 4
        
        q_value = self.model[index, action]

        #print("old q", self.model[index, action])

        next_index, next_rotation = get_state_index(next_state)
        next_value = np.max(self.model[next_index]) 

        new_q_value = (1 - ALPHA) * q_value + ALPHA * (reward + GAMMA * next_value)
        
        # Update Q-table
        self.model[index, action] = new_q_value

        #print("new q", self.model[index, action])
        
        # Or SARSA (On-Policy algorithm for TD-Learning) ?
        """the maximum reward for the next state is not necessarily used for updating the Q-values.
        Instead, a new action, and therefore reward, is selected using the same policy (eg e-greedy) that determined the original action.
        
        if random.uniform(0, 1) < epsilon:
            next_action = ACTIONS.index(np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1]))
        else:
            next_action = np.argmax(self.model[index])
            
        next_value = self.model[next_index, next_action]

        new_q_value = (1 - ALPHA) * q_value + ALPHA * (reward + GAMMA * next_value)
        
        self.model[index, action] = new_q_value

        """

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #print(f"End : {events}")
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    np.save("my-saved-model", self.model)
    
    #print(len(np.unique(self.model, axis = 0)))
    #with open("my-saved-model.pt", "wb") as file:
    #    pickle.dump(self.model, file)

    self.logger.info(f'End of round {last_game_state["round"]} with total reward : {self.rewards[-1]}')
    #print(f"Round {last_game_state['round']} : {self.rewards[-1]}")

    # Store training report
    #date = datetime.now().strftime("%d-%m-%Y")
    #file = "training_report_" + date
    
    #with open(file, 'a') as f:
    #    f.write(f"[{datetime.now().strftime('%d/%m %H_%M_%S')}] finished round {last_game_state['round']} with total reward {self.rewards[-1]} \n")

    #np.save(f"q_agent/rewards_{datetime.now().strftime('%d/%m %H_%M')}.npy", self.rewards[-1])


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.INVALID_ACTION: -100,
        e.RUN_IN_LOOP:-50,
        e.MOVED_UP:-2,
        e.MOVED_DOWN: -2,
        e.MOVED_LEFT: -2,
        e.MOVED_RIGHT: -2,
        e.WAITED: -20,
        e.COIN_COLLECTED: 100,
        e.COIN_DISTANCE_REDUCED: 10,
        e.COIN_DISTANCE_INCREASED: -5,
        e.BOMB_AVOIDED : 5,
        e.BOMB_DROPPED: -70,
        e.KILLED_OPPONENT: 500,
        e.GOT_KILLED: -100,
        e.KILLED_SELF: -150,
        e.SURVIVED_ROUND: 2,
        e.CRATE_DESTROYED: 20,
        e.COIN_FOUND: 15
        #e.BOMB_EXPLODED: ?
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    # update rewards
    self.rewards.append(reward_sum)
    
    return reward_sum
