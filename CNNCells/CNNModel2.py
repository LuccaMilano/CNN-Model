import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pdb import set_trace as pause

def CNNModel(input):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(3, 5, activation='linear', input_shape=(100,1)),
    tf.keras.layers.BatchNormalization( axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
    beta_constraint=None, gamma_constraint=None, synchronized=False),
    tf.keras.layers.Activation('elu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout( 0.2, noise_shape=None, seed=None),
    tf.keras.layers.Conv1D(3, 5, activation='linear'),
    tf.keras.layers.BatchNormalization( axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
    beta_constraint=None, gamma_constraint=None, synchronized=False),
    tf.keras.layers.Activation('elu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout( 0.2, noise_shape=None, seed=None),
    tf.keras.layers.Conv1D(3, 5, activation='linear'),
    tf.keras.layers.BatchNormalization( axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
    beta_constraint=None, gamma_constraint=None, synchronized=False),
    tf.keras.layers.Activation('elu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout( 0.2, noise_shape=None, seed=None),
    tf.keras.layers.Flatten(),

    ])

    model.summary()

    return model