from CNNCells import CNNModel
from mneExtraction import EEGExtract
import tensorflow as tf
from pdb import set_trace as pause
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


if __name__ == "__main__":
    mypath = "sleep-cassette"

    allFiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    eventsArray, dataArray = [], []
    for i in range (0,4,2): #23 patients
        epochs_train = EEGExtract.EEGClass("sleep-cassette/" + allFiles[i], "sleep-cassette/" + allFiles[i+1])
        eventsArray.append(epochs_train.epochs.events[:, 2])
        dataArray.append(epochs_train.epochs.get_data())
    
    # Array filtering
    dataArrayFiltered, eventsArrayFiltered = [], []
    for i in range(0,2):
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

    vmin = -70
    vmax = 70

    time = np.arange(0, 1, 1/100)
    for i in range(0, len(D)):
        if D[i] == 0:
            print("Acordado")
            C[i] *= 1e6
            plt.figure(figsize=(12, 2))
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
            })
            plt.plot(time, C[i])
            plt.xlabel('Tempo (s)')
            plt.ylabel('Tensão (µV)')
            plt.ylim([vmin, vmax])
            plt.tight_layout()
            plt.savefig("alert4.pdf",bbox_inches='tight')
            plt.show()
        if D[i] == 1:
            print("Sonolência")
            C[i] *= 1e6
            plt.figure(figsize=(12, 2))
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
            })
            plt.plot(time, C[i])
            plt.xlabel('Tempo (s)')
            plt.ylabel('Tensão (µV)')
            plt.ylim([vmin, vmax])
            plt.tight_layout()
            plt.savefig("drowsy4.pdf",bbox_inches='tight')
            plt.show()

    #    # Array filtering
    # dataArrayFiltered, eventsArrayFiltered = [], []
    # for i in range(0,10):
    #     random_arrays = dataArray[i][:4000]
    #     random_events = eventsArray[i][:4000]
    #     dataArrayFiltered.append(random_arrays)
    #     eventsArrayFiltered.append(random_events)
    
    # dataArrayFiltered = [item[:][:, 1, :] for item in dataArrayFiltered]
    # A = np.array(dataArrayFiltered)
    # B = np.array(eventsArrayFiltered)
    # C = A.reshape(-1, A.shape[-1])
    # D = B.reshape(-1)


    # CNN Model creation
    model = CNNModel.CNNModel()

    # Splitting the data
    train_ratio = 0.7
    test_ratio = 0.15
    val_ratio = 0.15
    train_data, temp_data, train_labels, temp_labels = train_test_split(C, D, test_size=(1 - train_ratio))
    test_data, val_data, test_labels, val_labels = train_test_split(temp_data, temp_labels, test_size=(val_ratio / (test_ratio + val_ratio)))


    # Define the hyperparameters for Adam optimizer
    #learning_rate = 1.2341e-07
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
    #model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=initial_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    history = model.fit(
        x=train_data,
        y=train_labels,
        batch_size=64,
        epochs=100,
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

    test_loss, test_accuracy = model.evaluate(test_data, test_labels)


