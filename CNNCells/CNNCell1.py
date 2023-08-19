import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pdb import set_trace as pause

def CNNModel(input):
    x = tf.keras.layers.Conv1D(9, 3, activation='linear')(input)
    x = tf.keras.layers.BatchNormalization( axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
    beta_constraint=None, gamma_constraint=None, synchronized=False)(x)
    x = tf.keras.layers.Activation('elu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout( 0.2, noise_shape=None, seed=None)(x)
    x = tf.keras.layers.Conv1D(9, 3, activation='linear')(x)
    x = tf.keras.layers.BatchNormalization( axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
    beta_constraint=None, gamma_constraint=None, synchronized=False)(x)
    x = tf.keras.layers.Activation('elu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout( 0.2, noise_shape=None, seed=None)(x)
    x = tf.keras.layers.Conv1D(9, 3, activation='linear')(x)
    x = tf.keras.layers.BatchNormalization( axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
    beta_constraint=None, gamma_constraint=None, synchronized=False)(x)
    x = tf.keras.layers.Activation('elu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout( 0.2, noise_shape=None, seed=None)(x)
    x = tf.keras.layers.Conv1D(9, 3, activation='linear')(x)
    x = tf.keras.layers.BatchNormalization( axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
    beta_constraint=None, gamma_constraint=None, synchronized=False)(x)
    x = tf.keras.layers.Activation('elu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout( 0.2, noise_shape=None, seed=None)(x)
    x = tf.keras.layers.Flatten()(x)
    return x