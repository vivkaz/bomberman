import tensorflow as tf
import pickle

import events as e

n_outputs = 6
inputs_shape = (5, 5, 3)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(20, 3, activation="elu", padding='same', input_shape=inputs_shape),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(20, 3, activation="elu", padding='same'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(20, activation="relu"),
    tf.keras.layers.Dense(n_outputs, activation="softmax")

])
model.save("initialize_model")
model.summary()

#Hyperparameter
Hyperparameter = {
"save_name" : "saved_model",
"epsilon_scale" : 700,
"learning_rate" : 1e-3,
"batch_size" : 30,
"steps" : 100,
"episoden" : 2000,
"discount_factor" : 0.9,
"rewards" : {
        e.INVALID_ACTION: -10,
        e.MOVED_UP: 2,
        e.MOVED_DOWN: 2,
        e.MOVED_LEFT: 2,
        e.WAITED: -10,
        e.MOVED_RIGHT: 2,
        #e.COIN_COLLECTED: 30,
        #e.COIN_DISTANCE_REDUCED: 5,
        #e.COIN_DISTANCE_INCREASED: -5,
        e.RUN_IN_LOOP:-10,
        #e.CRATE_DESTROYED: 40,
        e.BOMB_AVOIDED : 30,
        e.SURVIVED_ROUND : 5,
        e.KILLED_SELF: -50,
        e.IN_DANGER: -20,
        #e.COIN_FOUND : 20,
        e.GOT_KILLED : -50,
        e.BOMB_DROPPED : 10
},
"coin_density" : 0,#25
"crate_density" : 0.2,#0.5
"feature_setup" : {"feature_function" : "fake_coin_field_bombs",
                   "INPUTS" : [2,inputs_shape]}


}

with open('initialize_model/Hyperparameter.pkl', 'wb') as f:
    pickle.dump(Hyperparameter, f)
print("[info] model and hyperparameter file initialized")

