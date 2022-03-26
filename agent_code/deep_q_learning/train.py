from collections import namedtuple, deque
import tensorflow as tf
import pickle
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features
import matplotlib.pyplot as plt
from datetime import datetime
import time

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
    #print("länger replay_memory : ", len(self.replay_memory))
    indices = np.random.randint(len(self.replay_memory), size=batch_size)
    batch = [self.replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [np.array([experience[field_index] for experience in batch])
                                             for field_index in range(5)]
    return states, actions, rewards, next_states, dones


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # variable to buffer the experiences of previous setps
    self.replay_memory = deque(maxlen=1000)

    # variavle to save the rewards per step
    self.rewards = []
    self.loss = []

    self.buffer_states = deque(maxlen=6)

    self.training_history = deque(maxlen = 6000)

    #setup target model:
    if self.Hyperparameter["train_method"]["algo"] == "double_DQN":
        self.target = tf.keras.models.clone_model(self.model)
        self.target.set_weights(self.model.get_weights())

    self.visited_crates = []


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
    start_time = time.time()

    # function "buffer_information" saves the experience of one step in the variable self.replay_memory starts a step 2 since in setp 1 the old_game_state is none

    ###

    def update_buffer_states():
        if new_game_state is not None:
            self.buffer_states.append(np.array(new_game_state['self'][3]))  # save position as array not as tuple

    def risky_area(pos):
        #vertical = np.array([[x, pos[1]] for x in range(pos[0] - 3, pos[0] + 4)])
        #horizontal = np.array([[pos[0], y] for y in range(pos[1] - 3, pos[1] + 4)])
        buffer = [pos]
        pos = np.array([pos[0],pos[1]])
        for direction in [np.array([0,1]),np.array([0,-1]),np.array([1,0]),np.array([-1,0])]:
            current_position = np.copy(pos)
            for i in range(3):
                current_position += direction

                if old_game_state["field"][current_position[0],current_position[1]] == -1:
                    break
                else:
                    buffer.append(np.copy(current_position))


        buffer = np.array(buffer)
        return buffer

    def in_danger(area, bombs):
        danger = False
        #print(area.size,bombs.size)
        if area.size != 0 and bombs.size != 0:
            for b in bombs:
                for a in area:
                    if (b == a).all():
                    #if (np.array(b[0]) == a).all():
                        danger = True
                        break
        else:
            danger = False
        return danger

    def in_danger_with_timer(area, bombs, timer):
        danger = False
        bombs = bombs[timer < 3]
        for b in bombs:
            for a in area:
                if (b == a).all():
                #if (np.array(b[0]) == a).all():
                    danger = True

                    break
        return danger

    def get_bombs(game_state):
        bombs = np.empty((2, len(game_state["bombs"])))
        if bombs.size != 0:
            for count, bomb in enumerate(game_state["bombs"]):
                bombs[0, count] = bomb[0][0]
                bombs[1, count] = bomb[0][1]
            bombs = bombs.transpose()
        return bombs
    bombs_old = get_bombs(old_game_state)
    bombs_new = get_bombs(new_game_state)
    print("bombs_old : ", bombs_old)
    print("bombs_new : ", bombs_new)

    position_old = np.array(old_game_state["self"][3])
    position_new = np.array(new_game_state["self"][3])

    field_old = old_game_state["field"]
    field_new = new_game_state["field"]

    obsticle_field_old = np.where(field_old != 0,1,0)
    obsticle_field_new = np.where(field_new != 0,1,0)

    #register bombs in the field where all obsticles are combined
    if bombs_old.size != 0:
        #print("condition : ", len(bombs_old))
        #print("bombs_old : ",bombs_old.transpose())
        bombs_x_old = bombs_old.transpose()[0]
        bombs_y_old = bombs_old.transpose()[1]
        obsticle_field_old[bombs_x_old.astype(int), bombs_y_old.astype(int)] += 1
    if bombs_new.size != 0:
        #print("condition : ", len(bombs_old))
        #print("bombs_old : ", bombs_old.transpose())
        bombs_x_new = bombs_new.transpose()[0]
        bombs_y_new = bombs_new.transpose()[1]
        obsticle_field_new[bombs_x_new.astype(int), bombs_y_new.astype(int)] += 1



    feature_old = state_to_features(self, old_game_state, mode="normal")
    #print(feature_old)


    feature_new = state_to_features(self, new_game_state, mode="next_state")
    #print(feature_new)

    def get_in_Danger():
        #print(position_new)
        #print(bombs)
        if self_action != "BOMB":
        #if True:
            area_old = risky_area(position_old)
            area_new = risky_area(position_new)
            if not in_danger(area_old,bombs_old) and in_danger(area_new,bombs_new):
                events.append(e.GET_IN_DANGER)

    get_in_Danger()

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


    def agent_is_in_danger():
        bombs = np.empty((2,len(old_game_state["bombs"])))
        timer = np.empty(len(old_game_state["bombs"]))
        if bombs.size != 0:
            for count,bomb in enumerate(old_game_state["bombs"]):
                bombs[0,count] = bomb[0][0]
                bombs[1,count] = bomb[0][1]
                timer[count] = bomb[1]
            bombs = bombs.transpose()
            return in_danger_with_timer(risky_area(new_game_state["self"][3]),bombs,timer)




    def get_collectable_coins():

        coins = np.empty((2, len(old_game_state['coins'])))  # old state!, changed to len instead of sum
        if coins.size != 0:
            for count, coin in enumerate(old_game_state[
                                             'coins']):  # the "coin" entrie of game_state as not attribute collectable(it already contains the
                # collectable coins)
                coins[0, count] = coin[0]
                coins[1, count] = coin[1]
        return coins, coins.size != 0

    def coin_distance_reduced():#Add radius for interesting coins
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

        if agent_is_in_danger():
            events.append(e.IN_DANGER)


        #if bomb_dropped():
        #    events.append(e.BOMB_DROPPED)

        collectable_coins, reduced_dist = coin_distance_reduced()

        if collectable_coins:
            if reduced_dist:
                events.append(e.COIN_DISTANCE_REDUCED)
            else:
                events.append(e.COIN_DISTANCE_INCREASED)

    if run_in_loop():
        events.append(e.RUN_IN_LOOP)


    #print(events)
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    #print(events)

    def reach_crate():#checked and corrected
        #if old_game_state["step"] == 1:#no reward for second visit, prevent infinity loops or stand still at crate
        crate_coord = np.where(old_game_state["field"] == 1)
        visitable_crates = np.array([crate_coord[0],crate_coord[1]]).transpose()

        agents_neigh = np.array([np.array(new_game_state["self"][3]) + a for a in [np.array([0,1]),np.array([1,0]),np.array([0,-1]),np.array([-1,0])]])
        #print(agents_neigh)
        index = list(range(0,len(visitable_crates),1))
        for j in agents_neigh:
            for n,i in enumerate(visitable_crates):
                if (i == j).all():
                    if len(self.visited_crates) != 0:
                        if np.sum(np.sum(i == np.array(self.visited_crates),axis = 1) == 2) == 0:
                    #print(i,j)
                            events.append(e.CRATE_REACHED)#event occured
                            self.visited_crates.append(i)
                        else:
                            break
                    else:
                        events.append(e.CRATE_REACHED)  # event occured
                        self.visited_crates.append(i)

                    #index.remove(n)

        #self.visitable_crates = self.visitable_crates[index]
        #print(self.visitable_crates)




    reach_crate()



    def bomb_distance_increased():#checked and corrected
        def distance(point_1, point_2):
            return np.sqrt(np.power(point_1[0] - point_2[0], 2) + np.power(point_1[1] - point_2[1], 2))

        bombs = np.empty((2, len(old_game_state["bombs"])))
        d_old = []
        d_new = []
        if bombs.size != 0:
            for count, bomb in enumerate(old_game_state["bombs"]):
                bombs[0, count] = bomb[0][0]
                bombs[1, count] = bomb[0][1]

                d_old.append(distance(position_old,np.array([bomb[0][0],bomb[0][1]])))
                d_new.append(distance(position_new,np.array([bomb[0][0],bomb[0][1]])))
            bombs = bombs.transpose()
        bomb_sort = np.argsort(np.array(d_old))
        area = risky_area(position_old)

        for n,i in enumerate(bomb_sort):
            flag = False
            for j in area:
                if (j == bombs[i]).all():
                    flag = True
                    break

            #print("i : ", i, "n : ",n)
            #print("d_old[i] : ",d_old[i], "d_new[i] : ",d_new[i], "flag : ", flag)
            #print("d_old : ",d_old)
            #print("n_new : ",d_new)
            #print("bombs : ",bombs)
            if d_old[i] < d_new[i] and flag:
                events.append(e.BOMB_DISTANCE_INCREASED)
                break

    bomb_distance_increased()

    def get_trapped():
        view_range = self.Hyperparameter["feature_setup"]["INPUTS"][0]

        def neighbors(feature):
            position = np.array([view_range, view_range])
            field = feature[:, :, 0]#field with crates, opponents(not implemented), walls and bombs
            #print(feature[:, :, 2])
            explosion_field = np.where(feature[:,:,2] >= 5, 2,0)#field with active explosions

            A = []
            for i in [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]:
                A.append(position + i)
            A = np.array(A).transpose()
            return field[A[0], A[1]]+explosion_field[A[0],A[1]]

        neighbors_old = neighbors(feature_old)
        neighbors_new = neighbors(feature_new)
        block_values = [-1, -2, 1,2]  # values on field, that can can trap the agent -1 = wall, -2 = bomb, 1 = crate, 2 = active explosion
        block_status_old = 0
        block_status_new = 0
        for n in range(len(neighbors_new)):
            if (neighbors_old[n] in block_values):
                block_status_old += 1
            if (neighbors_new[n] in block_values):
                block_status_new += 1
        print(block_status_old,block_status_new)
        if block_status_new == 4 and block_status_old <= 3:
            return events.append(e.GET_TRAPPED)
    get_trapped()








    #change the case when there is bomb or free tile
    def certain_death():
        aef_old = self.advanced_explosion_field_old
        aef_new = self.advanced_explosion_field_new
        field_size = self.Hyperparameter["field_size"]

        def get_1d(array):
            x = array.transpose()[0]
            y = array.transpose()[1]
            return x * field_size + y

        def get_2d(array):
            x = np.trunc(array/field_size)
            y = array-x*field_size
            return np.stack((x,y)).transpose()
        def in_field(array):#only neighbors, which are in the playable field are considered
            lower_border = 1
            upper_border = field_size-2
            x = array.transpose()[0]
            y = array.transpose()[1]
            index = (x >= lower_border) & (y >= lower_border) & (x <= upper_border) & (y <= upper_border)
            return array[index]

        def get_explosion_neighbors(explosion_field,position):
            (coord_x, coord_y) = np.where((explosion_field > 0) & (explosion_field <=4))
            coord = np.array([coord_x,coord_y]).transpose()#all coordniates where a bomb is ticking
            print("coord : ", coord)
            if coord.size != 0:
                #discard all coordinates, which are not relevant for angents position
                d = np.sqrt(np.sum(np.power(coord - position,2),axis = 1))
                relevant_bombs = coord[d < 5]#thresold of relevant coordinates is set to 6 fields distance to our agent, reduce computational time
                L = len(relevant_bombs)
                a1 = np.array([1,0])
                a2 = np.array([0,1])
                A1 = np.broadcast_to(a1, (1,L , 2))
                A2 = np.broadcast_to(a2, (1, L, 2))
                F = np.concatenate((A1, A2, -A1, -A2),axis = 0)
                coord_neigh = np.broadcast_to(relevant_bombs,(4,L,2))-F
                coord_neigh = np.reshape(coord_neigh,(4*L,2))
                coord_neigh = in_field(np.unique(coord_neigh,axis = 0))
                coord_neigh_1d = get_1d(coord_neigh)
                relevant_bombs_1d = get_1d(relevant_bombs)
                coord_real_neigh = get_2d(np.setdiff1d(coord_neigh_1d, relevant_bombs_1d))
                return coord_real_neigh
            else:
                return np.array([])
        def get_free_neighbors(explosion_neighbors,field):
            if explosion_neighbors.size != 0:
                x_exp_neigh = explosion_neighbors[:,0].astype(int)
                y_exp_neigh = explosion_neighbors[:,1].astype(int)
                exp_neigh_values = field[x_exp_neigh,y_exp_neigh] == 0
                return explosion_neighbors[exp_neigh_values]
            else:
                return np.array([])

        def get_available_neighboors(position, tiles,field):
            if tiles.size != 0:
                necessary_steps = np.absolute(tiles[:,0]-position[0])+np.absolute(tiles[:,1]-position[1])
                bomb_timer = field[position[0],position[1]]
                return tiles[necessary_steps <= (5-bomb_timer)]
            else:
                return np.array([])
        def get_reachable_neighbors(position,tiles,field):
            #turns = [tiles[:,0]-position[0],[tiles[:,1]-position[1]]]
            #moves_vec = np.array([[0,1],[0,-1],[1,0],[-1,0]])
            #pos_vec = np.vstack((position,position,position,position))
            #paths = np.zeros((4,4,2))
            event = True
            if tiles.size != 0:
                for n,escape in enumerate(tiles):
                    x_diff = escape[0]-position[0]
                    y_diff = escape[1]-position[1]
                    if np.absolute(x_diff) > np.absolute(y_diff):
                        move_1 = np.broadcast_to(np.sign(x_diff)*np.array([1,0]),(np.absolute(x_diff).astype(int),2))
                        move_2 = np.broadcast_to(np.sign(y_diff)*np.array([0,1]),(np.absolute(y_diff).astype(int),2))
                    elif np.absolute(x_diff) < np.absolute(y_diff):
                        move_1 = np.broadcast_to(np.sign(y_diff) * np.array([0, 1]), (np.absolute(y_diff).astype(int), 2))
                        move_2 = np.broadcast_to(np.sign(x_diff) * np.array([1, 0]), (np.absolute(x_diff).astype(int), 2))
                    if np.absolute(x_diff) == np.absolute(y_diff) and np.absolute(x_diff) == 1:
                        move_1 = np.broadcast_to(np.sign(x_diff)*np.array([1,0]),(np.absolute(x_diff).astype(int),2))
                        move_2 = np.broadcast_to(np.sign(y_diff)*np.array([0,1]),(np.absolute(y_diff).astype(int),2))
                        print("position + move_1",position + move_1)
                        index = (position+move_1).transpose()
                        print("index : ", index)
                        #print("1 step situation first try : ",field[int(index[0]),int(index[1])])
                        if field[int(index[0]),int(index[1])] != 0:
                            #print("1 step situation second try")
                            move_1 = np.broadcast_to(np.sign(y_diff)*np.array([0,1]),(np.absolute(y_diff).astype(int),2))
                            move_2 = np.broadcast_to(np.sign(x_diff)*np.array([1,0]),(np.absolute(x_diff).astype(int),2))
                    path = np.cumsum(np.vstack((position,move_1,move_2)),axis = 0)
                    #print(f"path {path} for escape {escape} from position {position}")
                    x_path = path.transpose()[0][1:]
                    y_path = path.transpose()[1][1:]

                    print("path_value = ",np.sum(field[x_path.astype(int),y_path.astype(int)]))
                    if np.sum(field[x_path.astype(int),y_path.astype(int)]) == 0:
                        print("escape_path : ",path)
                        event = False
                        break

            else:
                event = True
            return event



        def get_certain_death(position,aef,obsticle_field):
            if aef[int(position[0]), int(position[1])] != 0:  # find save place only if there is an incoming explosion
                neigh = get_explosion_neighbors(aef, position)
                neigh_free = get_free_neighbors(neigh, obsticle_field)
                neigh_ava = get_available_neighboors(position, neigh_free, obsticle_field)
                #print("coord_real_neigh", neigh)
                #print("free_neighbors : ", neigh_free)
                print("in_range neighbors : ", neigh_ava)
                cer_death =  get_reachable_neighbors(position, neigh_ava, obsticle_field)
            else:
                cer_death = False
            return cer_death
        print("-----------old_state certain_death prints : ----------------")
        cer_death_old = get_certain_death(position_old,aef_old,obsticle_field_old)
        print("result _ old : ",cer_death_old)
        print("-----------new_state certain_death prints : ----------------")
        cer_death_new = get_certain_death(position_new,aef_new,obsticle_field_new)
        print("result _ new : ",cer_death_new)

        if cer_death_new and not cer_death_old:
            events.append(e.CERTAIN_DEATH)


    certain_death()
    #print("aef_old : ",self.advanced_explosion_field_old)
    #print("aef_new : ",self.advanced_explosion_field_new)









    def buffer_information():
        #print(f"old_game_state : ", old_game_state)
        #print(f"game_state : ", new_game_state)
        #print("train - buffer")

        done = False
        buffer_time_start = time.time()

        old_feature = state_to_features(self, old_game_state,mode = "normal")

        t_1 = time.time()

        new_feature = state_to_features(self, new_game_state,mode = "next_state")

        t_2 = time.time()
        self.replay_memory.append((old_feature, self_action,
                                       reward_from_events(self, events), new_feature,done))
        buffer_time_end = time.time()

        #times = np.array([buffer_time_end, buffer_time_end,t_2, t_1]) - np.array(
        #    [buffer_time_start,t_2, t_1, buffer_time_start])
        #print("buffer time = ", np.round(times, 9))
        #print(f"events occured buffered :  {events} \n"
        #      f"in step {new_game_state['step']}")
        #print(f"action : {self_action} at step {new_game_state['step']}")





    buffer_information()
    self.rewards.append(reward_from_events(self,events))

    print(events)
    #print(reward_from_events(self,events))

    end_time = time.time()
    #times = np.array([end_time,buffer_time_end, t_2, t_1])-np.array([start_time,buffer_time_start ,t_1,buffer_time_start])
    #print("train : game_events_occured time = ", np.round(times,9))

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
    start_time = time.time()

    #add last action to buffer
    done = True
    self.replay_memory.append((state_to_features(self,last_game_state),last_action,
                        reward_from_events(self,events),state_to_features(self,last_game_state), done))
    self.rewards.append(reward_from_events(self,events))

    #print("end of round : ",events)

    batch_size = self.Hyperparameter["batch_size"]  # number of old experience used for updating the model
    discount_factor = self.Hyperparameter["discount_factor"]  # value which weights
    optimizer = tf.keras.optimizers.Adam(learning_rate=self.Hyperparameter["learning_rate"])
    loss_fn = tf.keras.losses.mean_squared_error



#normal DQN
    def DQN(input):
        if last_game_state["round"] >= input[0]:
            experiences = sample_experiences(self, batch_size)
            states, actions, rewards, next_states, dones = experiences
            states = np.array(states)
            next_states = np.array(next_states)

            next_Q_values = self.model.predict(next_states)
            max_next_Q_values = np.max(next_Q_values, axis=1)
            target_Q_values = (rewards + (1-dones) * discount_factor * max_next_Q_values)
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
                self.loss.append(loss)

            grads = tape.gradient(loss, self.model.trainable_variables)
            # print("grads: ",grads)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            self.logger.info("update model")
#double DQN
    def Double_DQN(input):
        if last_game_state["round"] >= input[0]:
            experiences = sample_experiences(self, batch_size)
            states, actions, rewards, next_states, dones = experiences
            states = np.array(states)
            next_states = np.array(next_states)

            next_Q_values = self.model.predict(next_states)#
            best_next_action = np.argmax(next_Q_values, axis=1)
            next_mask = tf.one_hot(best_next_action, self.n_outputs).numpy()

            next_best_Q_values = (self.target.predict(next_states)*next_mask).sum(axis = 1)

            target_Q_values = (rewards + (1-dones) * discount_factor * next_best_Q_values)
            target_Q_values = target_Q_values.reshape(-1, 1)

            actions = transform_actions_to_number(actions)
            mask = tf.one_hot(actions, self.n_outputs)
            with tf.GradientTape() as tape:
                # print("states",states)
                # print(self.model(states))
                all_Q_values = self.model(states)
                Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
                loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
                self.loss.append(loss)

            grads = tape.gradient(loss, self.model.trainable_variables)
            # print("grads: ",grads)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            self.logger.info("update model")
        if last_game_state["round"] %  input[1] == 0:
            self.target.set_weights(self.model.get_weights())

    if self.Hyperparameter["train_method"]["algo"] == "DQN":
        DQN(self.Hyperparameter["train_method"]["INPUTS"])
    elif self.Hyperparameter["train_method"]["algo"] == "double_DQN":
        Double_DQN(self.Hyperparameter["train_method"]["INPUTS"])


    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Store the model
    # with open("my-saved-model.pt", "wb") as file:
    #    pickle.dump(self.model, file)
    sum = np.sum(self.rewards)

    # self.rewards.append(np.sum(self.replay_memory[3]))

    #self.replay_memory.clear()
    self.logger.info(f'end of round {last_game_state["round"]} with total reward : {sum}')

    self.training_history.append(sum)




    # reset protokoll file in first round
    """
    if last_game_state["round"] == 1:
        with open("training_protokoll.txt", 'r+') as f:
            f.truncate(0)

    # write data to protokoll such that the training status can be supervised during training
    with open("training_protokoll.txt", 'a') as f:
        f.write(
            f"[{datetime.now().strftime('%H_%M_%S')}] finished epoche {last_game_state['round']} with total reward {sum} , {self.rewards}")
        f.write("\n")
    """
        # save model and training_history
    if last_game_state["round"] == 1:
        now = datetime.now()
        self.date_time = now.strftime("%m_%d_%Y_%H_%M_%S")

    if self.Hyperparameter["episoden"] <= last_game_state["round"]+1:
        print("save_model")
        print("DQN",self.Hyperparameter["train_method"]["algo"])
        print(type("DQN"), type(self.Hyperparameter["train_method"]["algo"]))
        if self.Hyperparameter["train_method"]["algo"] == "DQN":
            self.model.save(self.Hyperparameter["save_name"])
            print(self.Hyperparameter['save_name'])
            print(f"[info] model saved under name {self.Hyperparameter['save_name']}")
        elif self.Hyperparameter["train_method"]["algo"] == "double_DQN":
            self.target.save(self.Hyperparameter["save_name"])
            print(f"[info] model saved under name {self.Hyperparameter['save_name']}")

        with open(f'{self.Hyperparameter["save_name"]}/Hyperparameter.pkl', 'wb') as f:
            pickle.dump(self.Hyperparameter, f)

        np.save(f"{self.Hyperparameter['save_name']}/rewards_{self.date_time}.npy", self.training_history)
    self.rewards.clear()
    end_time = time.time()
    #print("end of round : time = ",np.round(end_time-start_time,5))
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
