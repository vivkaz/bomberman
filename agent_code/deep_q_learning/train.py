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

#function to give an action like "WAIT" a specific number
def transform_actions_to_number(action):
    result = []
    for act in action:
        result.append(ACTIONS.index(act))
    return np.array(result)

#function "sample_experiences" samples a random set of experiences. A experience is the information about the state the action the reward and the next state of one step in the game.
#the parameter batch_size defines the number of experiences that are use for one training iteration
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

    #variable to buffer the experiences of previous setps
    self.replay_memory = deque(maxlen=2000)

    #variavle to save the rewards per step
    self.rewards = []


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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    #function "buffer_information" saves the experience of one step in the variable self.replay_memory starts a step 2 since in setp 1 the old_game_state is none
    def buffer_information():
        if new_game_state["step"] >=2:
            self.replay_memory.append((state_to_features(old_game_state),self_action,reward_from_events(self,events),state_to_features(new_game_state)))
    self.old_game_state = old_game_state
    self.new_game_state = new_game_state

    buffer_information()



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

    batch_size = 50#number of old experience used for updating the model
    discount_factor = 0.99#value which weights
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    loss_fn = tf.keras.losses.mean_squared_error

    if last_game_state["round"] >= 1:
        experiences = sample_experiences(self, batch_size)
        states, actions, rewards, next_states = experiences
        states = np.array(states)
        next_states = np.array(next_states)

        with open("states.npy", 'wb') as f:
            np.save(f, states, allow_pickle=True)
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
        #print("grads: ",grads)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.logger.info("update model")

    #save model
    self.model.save("saved_model")


    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')


    # Store the model
    #with open("my-saved-model.pt", "wb") as file:
    #    pickle.dump(self.model, file)
    sum = 0
    for i in range(len(self.replay_memory)):
        sum += self.replay_memory[i][2]
    #self.rewards.append(np.sum(self.replay_memory[3]))
    self.replay_memory.clear()
    self.rewards.append(sum)
    self.logger.info(f'end of round {last_game_state["round"]} with total reward : {sum}')

    if last_game_state["round"] ==1:
        now = datetime.now()
        self.date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    with open(f"rewards_{self.date_time}.txt",'a') as f:
        f.write(f'{sum}')
        f.write('\n')

    #reset protokoll file in first round
    if last_game_state["round"] == 1:
        with open("training_protokoll.txt",'r+') as f:
            f.truncate(0)

    #write data to protokoll such that the training status can be supervised during training
    with open("training_protokoll.txt",'a') as f:
        f.write(f"[{datetime.now().strftime('%H_%M_%S')}] finished epoche {last_game_state['round']} with total reward {sum}")
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
        d = np.max(np.sqrt(np.power(position[0]-x_coin,2)+np.power(position[1]-y_coin,2)))
        return d


    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """


    game_rewards = {
        e.INVALID_ACTION: -10,
        e.MOVED_UP:-2,
        e.MOVED_DOWN: -2,
        e.MOVED_LEFT: -2,
        e.WAITED: -5,
        e.MOVED_RIGHT: -2,
        e.COIN_COLLECTED: 20,
        e.COIN_DISTANCE_REDUCED: 5,
        e.COIN_DISTANCE_INCREASED: -5,
        e.INFINITY_LOOP:-20
        #e.AVOID_BOMB: 1


    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
