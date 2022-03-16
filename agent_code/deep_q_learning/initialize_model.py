import tensorflow as tf
import pickle

import events as e

n_outputs = 5
inputs_shape = (7, 7, 2)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(20, 3, activation="elu", padding='same', input_shape=inputs_shape),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(20, 3, activation="elu", padding='same'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(n_outputs, activation="softmax")

])
model.save("initialize_model")
model.summary()

#Hyperparameter
Hyperparameter = {
"save_name" : "saved_model",
"epsilon_scale" : 1,
"learning_rate" : 1e-2,
"batch_size" : 50,
"steps" : 100,
"episoden" : 1,
"discount_factor" : 0.9,
"rewards" : {
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
},
"coin_density" : 9,
"crate_density" : 0,
"feature_setup" : {"feature_function" : "field_coin_map",
                   "INPUTS" : [3,inputs_shape,False]}


}

with open('initialize_model/Hyperparameter.pkl', 'wb') as f:
    pickle.dump(Hyperparameter, f)
print("[info] model and hyperparameter file initialized")

