from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from pdb import set_trace as pause
import tensorflow as tf
import os
import numpy as np
from mneExtraction import EEGExtract
from Wavelet.haar import *
from CNNCells import CNNModel1, CNNModel2

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


if __name__ == "__main__":
    mypath = "sleep-cassette"

    allFiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    eventsArray, dataArray = [], []
    for i in range (0,20,2): #23 patients
        epochs_train = EEGExtract.EEGClass("sleep-cassette/" + allFiles[i], "sleep-cassette/" + allFiles[i+1])
        eventsArray.append(epochs_train.epochs.events[:, 2])
        dataArray.append(epochs_train.epochs.get_data())
    

    # Array filtering
    dataArrayFiltered, eventsArrayFiltered = [], []
    for i in range(0,10):
        random_arrays = dataArray[i][:4000]
        random_events = eventsArray[i][:4000]
        dataArrayFiltered.append(random_arrays)
        eventsArrayFiltered.append(random_events)
    
    dataArrayFiltered = [item[:][:, 1, :] for item in dataArrayFiltered]
    A = np.array(dataArrayFiltered)
    B = np.array(eventsArrayFiltered)
    C = A.reshape(-1, A.shape[-1])
    D = B.reshape(-1)


    # CNN Model creation
    model1 = CNNModel1.CNNModel(input)
    model2 = CNNModel2.CNNModel(input)
    
    input = tf.keras.Input(shape=(100,1), name='input')
    models = [model1, model2]
    outputs = [model(input) for model in models]
    x = tf.keras.layers.Concatenate()(outputs)

    output = tf.keras.layers.Dense(20, activation='relu')(x) 
    output = tf.keras.layers.Dropout( 0.2, noise_shape=None, seed=None)(output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)
    conc_model = tf.keras.Model(input, output, name= 'Concatenated_Model')
    conc_model.summary()

    # Splitting the data
    train_ratio = 0.7
    test_ratio = 0.15
    val_ratio = 0.15
    train_data, temp_data, train_labels, temp_labels = train_test_split(C, D, test_size=(1 - train_ratio))
    test_data, val_data, test_labels, val_labels = train_test_split(temp_data, temp_labels, test_size=(val_ratio / (test_ratio + val_ratio)))


    # Define the hyperparameters for Adam optimizer
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-08

    # Create the Adam optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )

    initial_learning_rate = 0.001
    conc_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=initial_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    #conc_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    history = conc_model.fit(
        x=train_data,
        y=train_labels,
        batch_size=8,
        epochs=20,
        callbacks=[callback],
        validation_data=(val_data, val_labels)
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plotting the training accuracy and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


