
import tensorflow as tf


class Brain(object):
  def __init__(self, lr = 0.0001, num_actions = 3, num_inputs = 5):
    self.learning_rate = lr

    states = tf.keras.layers.Input(shape = (num_inputs, ))
    x = tf.keras.layers.Dense(units = 64, activation = 'sigmoid')(states)
    x = tf.keras.layers.Dropout(rate = 0.1)(x)
    y = tf.keras.layers.Dense(units = 32, activation = 'sigmoid')(x)
    y = tf.keras.layers.Dropout(rate = 0.1)(y)
    q_vals = tf.keras.layers.Dense(units = num_actions, activation = 'softmax')(y)
    self.model = tf.keras.models.Model(inputs = states, outputs = q_vals)
    self.model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(learning_rate = lr))

