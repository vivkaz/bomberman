import tensorflow as tf
import pickle

import events as e

n_outputs = 6
inputs_shape = (9, 9, 3)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(40, 4, activation="relu", padding='same', input_shape=inputs_shape),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(40, 3, activation="relu", padding='same'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(1, activation="relu"),
    #tf.keras.layers.Dense(200, activation="elu"),
    tf.keras.layers.Dense(70, activation="elu"),
    tf.keras.layers.Dense(30, activation="elu"),
    tf.keras.layers.Dense(n_outputs, activation="softmax")

])
model.save("initialize_model")
model.summary()

#Hyperparameter
Hyperparameter = {
"save_name" : "saved_model",#check, that the model name is not redefined in callbacks
"epsilon_scale" : 1000,#check if epsilon is not redefined in callbacks
"learning_rate" : 5e-4,
"batch_size" : 50,
"steps" : 50, #check steps in settings
"episoden" : 2000,
"discount_factor" : 0.8,
"n_outputs" : n_outputs,
"rewards" :
            {
        e.INVALID_ACTION: -20,
        e.MOVED_UP: -2,
        e.MOVED_DOWN: -2,
        e.MOVED_LEFT: -2,
        e.WAITED: -30,
        e.MOVED_RIGHT: -2,
        e.COIN_COLLECTED: 100,
        e.COIN_DISTANCE_REDUCED: 20,
        e.COIN_DISTANCE_INCREASED: -10,
        e.RUN_IN_LOOP:-10,
        e.CRATE_DESTROYED: 20,
        e.BOMB_AVOIDED : 100,
        e.SURVIVED_ROUND : 3,
        e.KILLED_SELF: -150,
        e.IN_DANGER: -5,
        e.COIN_FOUND : 15,
        e.GOT_KILLED : -50,
        e.BOMB_DROPPED : -5,
        e.CRATE_REACHED: 40,
        e.BOMB_DISTANCE_INCREASED: 50,
        e.GET_IN_DANGER: -30,
        e.CERTAIN_DEATH: -100,
        e.GET_TRAPPED: -100,
        e.KILLED_OPPONENT: 50
},
"coin_density" : 20,#25
"crate_density" : 0.7,#0.5
"field_size" : 17,#check in settings
"feature_setup" : {"feature_function" : "fake_coin_crate_field_bombs",
                   "INPUTS" : [4,inputs_shape]},

#"train_method" : {"algo": "PER_DQN","INPUTS" : [1,0.6,0.4,0.1] }
"train_method" : {"algo": "double_DQN", "INPUTS" : [10,2]}
#"train_method" : {"algo": "DQN", "INPUTS" : [10,10]}




}

with open('initialize_model/Hyperparameter.pkl', 'wb') as f:
    pickle.dump(Hyperparameter, f)
print("[info] model and hyperparameter file initialized")

""" {
        e.INVALID_ACTION: -10,
        e.MOVED_UP: -2,
        e.MOVED_DOWN: -2,
        e.MOVED_LEFT: -2,
        e.WAITED: -5,
        e.MOVED_RIGHT: -2,
        e.COIN_COLLECTED: 20,
        e.COIN_DISTANCE_REDUCED: 5,
        e.COIN_DISTANCE_INCREASED: -5,
        e.RUN_IN_LOOP:-10},
        """