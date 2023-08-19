from os import listdir
from os.path import isfile, join
from sklearn.model_selection import KFold
from pdb import set_trace as pause
import tensorflow as tf
import os
import numpy as np
from mneExtraction import EEGExtract
from Wavelet.haar import *
from CNNCells import CNNModel1, CNNModel2
 

if __name__ == "__main__":
    mypath = "sleep-cassette"

    allFiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    eventsArray, dataArray = [], []
    for i in range (0,46,2): #23 patients
        epochs_train = EEGExtract.EEGClass("sleep-cassette/" + allFiles[i], "sleep-cassette/" + allFiles[i+1])
        eventsArray.append(epochs_train.epochs.events[:, 2])
        dataArray.append(epochs_train.epochs.get_data())
    
    # Array filtering
    dataArrayFiltered, eventsArrayFiltered = [], []
    for i in range(0,23):
        indices_ones = np.where(eventsArray[i] == 1)[0]
        indices_zero = np.where(eventsArray[i] == 0)[0]

        # Randomly shuffle the indices
        np.random.shuffle(indices_ones)
        np.random.shuffle(indices_zero)

        # Select the first 100 indices for each value
        selected_indices_ones = indices_ones[:600]
        selected_indices_zero = indices_zero[:600]

        combined_indices = np.concatenate([selected_indices_ones, selected_indices_zero])
        np.random.shuffle(combined_indices)
        if len(combined_indices) == 1200:
            random_arrays = dataArray[i][combined_indices]
            random_events = eventsArray[i][combined_indices]
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

    output = tf.keras.layers.Dense(12, activation='relu')(x) 
    output = tf.keras.layers.Dropout( 0.2, noise_shape=None, seed=None)(output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)
    conc_model = tf.keras.Model(input, output, name= 'Concatenated_Model')
    conc_model.summary()

    # Define the hyperparameters for Adam optimizer
    learning_rate = 0.0001
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

    conc_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    num_folds = 5
    kf = KFold(n_splits=num_folds)

    for fold, (train_indices, val_indices) in enumerate(kf.split(D)):
        print(f"Fold {fold + 1}/{num_folds}")

        data_train = C[train_indices]
        labels_train = D[train_indices]

        data_val = C[val_indices]
        labels_val = D[val_indices]

        history = conc_model.fit(
            x=data_train,
            y=labels_train,
            batch_size=2,
            epochs=10,
            validation_data=(data_val, labels_val)
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


