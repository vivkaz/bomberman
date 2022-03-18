from collections import namedtuple, deque
import tensorflow as tf
import pickle
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features
import matplotlib.pyplot as plt
from datetime import datetime


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
def transform_actions_to_number(action):
    result = []
    for act in action:
        result.append(ACTIONS.index(act))
    return np.array(result)


def sample_experiences(self, batch_size):
    indices = np.random.randint(len(self.replay_memory), size=batch_size)
    batch = [self.replay_memory[index] for index in indices]
    #print(batch[0].dtype)

    states, actions, rewards, next_states = [np.array([experience[field_index] for experience in batch])
                                                    for field_index in range(4)]
    #print("states.dtype", states.dtype)

    #print("states",states)
    #print(type(states[0]))
    #print(type(states))
    return states, actions, rewards, next_states
def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.replay_memory = deque(maxlen=2000)
    self.rewards = []



    #function "sample_experience" returns the samples information from the buffer. The size is defined by the batch size



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
    def buffer_information():
        if new_game_state["step"] >=2:
            #print("state_to_features_buffer : ",type((state_to_features(old_game_state))))

                #print("Nonetype information : " ,state_to_features(old_game_state),old_game_state)
            self.replay_memory.append((state_to_features(old_game_state),self_action,reward_from_events(self,events),state_to_features(new_game_state)))
    # Idea: Add your own events to hand out rewards
    buffer_information()

    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


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

    batch_size = 32
    discount_factor = 0.95
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
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

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

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.INVALID_ACTION: -20,
        e.MOVED_UP:5,
        e.MOVED_DOWN: 5,
        e.MOVED_LEFT: 5,
        e.WAITED: -20,
        e.MOVED_RIGHT: 5,
        e.COIN_COLLECTED: 20

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
