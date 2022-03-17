from collections import namedtuple, deque
import tensorflow as tf
import pickle
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features
import matplotlib.pyplot as plt
from datetime import datetime

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


# function to give an action like "WAIT" a specific number
def transform_actions_to_number(action):
    result = []
    for act in action:
        result.append(ACTIONS.index(act))
    return np.array(result)


# function "sample_experiences" samples a random set of experiences. A experience is the information about the state the action the reward and the next state of one step in the game.
# the parameter batch_size defines the number of experiences that are use for one training iteration
def sample_experiences(self, batch_size):
    indices = np.random.randint(len(self.replay_memory), size=batch_size)
    batch = [self.replay_memory[index] for index in indices]
    states, actions, rewards, next_states = [np.array([experience[field_index] for experience in batch])
                                             for field_index in range(4)]
    return states, actions, rewards, next_states


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # variable to buffer the experiences of previous setps
    self.replay_memory = deque(maxlen=2000)

    # variavle to save the rewards per step
    self.rewards = []

    self.buffer_states = deque(maxlen=6)


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

    # function "buffer_information" saves the experience of one step in the variable self.replay_memory starts a step 2 since in setp 1 the old_game_state is none
    def buffer_information():
        if new_game_state["step"] >= 2:
            self.replay_memory.append((state_to_features(self, old_game_state), self_action,
                                       reward_from_events(self, events), state_to_features(self, new_game_state)))

    self.old_game_state = old_game_state
    self.new_game_state = new_game_state

    buffer_information()

    ###
    def update_buffer_states():
        if new_game_state is not None:
            self.buffer_states.append(np.array(new_game_state['self'][3]))  # save position as array not as tuple

    def risky_area(pos):
        vertical = np.array([[x, pos[1]] for x in range(pos[1] - 3, pos[1] + 4)])
        horizontal = np.array([[pos[0], y] for y in range(pos[0] - 3, pos[0] + 4)])
        return np.concatenate((vertical, horizontal), axis=0)

    def in_danger(area, bombs):
        danger = False
        for b in bombs:
            for a in area:
                if (b == a).all():
                    danger = True
                    break
        return danger

    def bomb_dropped():
        return old_game_state['self'][2] == True and new_game_state['self'][2] == False

    def bomb_avoided():
        old_risky_area = risky_area(old_game_state['self'][3])
        new_risky_area = risky_area(new_game_state['self'][3])

        bombs = np.empty((2, len(old_game_state['bombs'])))
        if bombs.size != 0:
            for count, bomb in enumerate(old_game_state['bombs']):
                bombs[0, count] = bomb[0][0]
                bombs[1, count] = bomb[0][1]
            bombs = bombs.transpose()
            old_danger = in_danger(old_risky_area, bombs)
            new_danger = in_danger(new_risky_area, bombs)
            return old_danger and not new_danger

    def get_collectable_coins():

        coins = np.empty((2, len(old_game_state['coins'])))  # old state!, changed to len instead of sum
        if coins.size != 0:
            for count, coin in enumerate(old_game_state[
                                             'coins']):  # the "coin" entrie of game_state as not attribute collectable(it already contains the
                # collectable coins)
                coins[0, count] = coin[0]
                coins[1, count] = coin[1]
        return coins, coins.size != 0

    def coin_distance_reduced():
        old_position = old_game_state['self'][3]
        new_position = new_game_state['self'][3]
        coins, flag = get_collectable_coins()

        if flag:  # coins not empty
            def dist(position, coins):
                return np.sqrt(np.power(coins[0] - position[0], 2) + np.power(coins[1] - position[1], 2))

            old_dist = dist(old_position, coins)
            new_dist = dist(new_position, coins)
            return flag, np.min(new_dist) < np.min(old_dist)
        else:
            return flag, False

    def run_in_loop():
        update_buffer_states()

        if len(self.buffer_states) == 6:
            buffer = np.array(self.buffer_states)
            return len(np.unique(buffer[[0, 2, 4]], axis=0)) == 1 and len(
                        np.unique(buffer[[1, 3, 5]], axis=0)) == 1 and (
                            np.unique(buffer[[0, 2, 4]],axis = 0)[0] != np.unique(buffer[[1, 3, 5]],axis = 0)[0]).any()
        else:
            return False

    # Add your own events to hand out rewards
    if old_game_state is not None:
        if bomb_avoided():
            events.append(e.BOMB_AVOIDED)

        if bomb_dropped():
            events.append(e.BOMB_DROPPED)

        collectable_coins, reduced_dist = coin_distance_reduced()
        if collectable_coins:
            if reduced_dist:
                events.append(e.COIN_DISTANCE_REDUCED)
            else:
                events.append(e.COIN_DISTANCE_INCREASED)

    if run_in_loop():
        print(self.buffer_states)
        events.append(e.RUN_IN_LOOP)

    #print(events)
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    #print(events)
    print(f"events occured :  {events} \n"
          f"in step {new_game_state['step']}")
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

    batch_size = self.Hyperparameter["batch_size"]  # number of old experience used for updating the model
    discount_factor = self.Hyperparameter["discount_factor"]  # value which weights
    optimizer = tf.keras.optimizers.Adam(learning_rate=self.Hyperparameter["learning_rate"])
    loss_fn = tf.keras.losses.mean_squared_error

    if last_game_state["round"] >= 1:
        experiences = sample_experiences(self, batch_size)
        states, actions, rewards, next_states = experiences
        states = np.array(states)
        next_states = np.array(next_states)

        next_Q_values = self.model.predict(next_states)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards + (1) * discount_factor * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)
        # print("actions : ",actions)
        actions = transform_actions_to_number(actions)
        # print("actions : ", actions)
        mask = tf.one_hot(actions, self.n_outputs)
        with tf.GradientTape() as tape:
            # print("states",states)
            # print(self.model(states))
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

        grads = tape.gradient(loss, self.model.trainable_variables)
        # print("grads: ",grads)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.logger.info("update model")

    # save model
    if self.Hyperparameter["episoden"] <= last_game_state["round"]:
        self.model.save(self.Hyperparameter["save_name"])
        with open(f'{self.Hyperparameter["save_name"]}/Hyperparameter.pkl', 'wb') as f:
            pickle.dump(self.Hyperparameter, f)

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Store the model
    # with open("my-saved-model.pt", "wb") as file:
    #    pickle.dump(self.model, file)
    sum = 0
    for i in range(len(self.replay_memory)):
        sum += self.replay_memory[i][2]
    # self.rewards.append(np.sum(self.replay_memory[3]))
    self.replay_memory.clear()
    self.rewards.append(sum)
    self.logger.info(f'end of round {last_game_state["round"]} with total reward : {sum}')

    if last_game_state["round"] == 1:
        now = datetime.now()
        self.date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    with open(f"rewards_{self.date_time}.txt", 'a') as f:
        f.write(f'{sum}')
        f.write('\n')

    # reset protokoll file in first round
    if last_game_state["round"] == 1:
        with open("training_protokoll.txt", 'r+') as f:
            f.truncate(0)

    # write data to protokoll such that the training status can be supervised during training
    with open("training_protokoll.txt", 'a') as f:
        f.write(
            f"[{datetime.now().strftime('%H_%M_%S')}] finished epoche {last_game_state['round']} with total reward {sum}")
        f.write("\n")


def reward_from_events(self, events: List[str]) -> int:
    def distance_to_nearest_coin(state):
        position = state["self"][3]
        coins_position = state["coins"]
        x_coin = []
        y_coin = []
        for i in coins_position:
            x_coin.append(i[0])
            y_coin.append(i[1])
        x_coin = np.array(x_coin)
        y_coin = np.array(y_coin)
        d = np.max(np.sqrt(np.power(position[0] - x_coin, 2) + np.power(position[1] - y_coin, 2)))
        return d

    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    game_rewards = self.Hyperparameter["rewards"]
    """
    game_rewards = {
        e.INVALID_ACTION: -10,
        e.MOVED_UP:-2,
        e.MOVED_DOWN: -2,
        e.MOVED_LEFT: -2,
        e.WAITED: -5,
        e.MOVED_RIGHT: -2,
        e.COIN_COLLECTED: 30,
        e.COIN_DISTANCE_REDUCED: 10,
        e.COIN_DISTANCE_INCREASED: -5,
        e.RUN_IN_LOOP:-20
        #e.BOMB_AVOIDED : 1


    }
    """
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
