from sklearn.model_selection import KFold
from pdb import set_trace as pause
import tensorflow as tf
import os
import sys
import numpy as np
from mneExtraction import EEGExtract
from CNNCells import CNNCell1
from Wavelet.haar import *
from sklearn.metrics import confusion_matrix
import seaborn as sns

if __name__ == "__main__":
    mypath = "sleep-cassette"

    allFiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    eventsArray, dataArray = [], []
    for i in range (0,46,2): #23 patients
        epochs_train = EEGExtract.EEGClass("sleep-cassette/" + allFiles[i], "sleep-cassette/" + allFiles[i+1])
        eventsArray.append(epochs_train.epochs.events[:, 2])
        dataArray.append(epochs_train.epochs.get_data())


    dataArray = [item[:][:, 1, :] for item in dataArray]
    
    patients_array = []
    for i in range(0, len(dataArray)):
        combined_array = []
        for j in range(0, len(dataArray[i])-1):
            combined_array.append(np.concatenate((dataArray[i][j], dataArray[i][j+1][:28])))
        combined_array.append(np.concatenate((dataArray[i][-1], dataArray[i][0][:28])))
        patients_array.append(np.array(combined_array))
   
    # Array filtering
    dataArrayFiltered, eventsArrayFiltered = [], []
    for i in range(0,23):
        indices_ones = np.where(eventsArray[i] == 1)[0]
        indices_zero = np.where(eventsArray[i] == 0)[0]

        # Randomly shuffle the indices
        np.random.shuffle(indices_ones)
        np.random.shuffle(indices_zero)

        # Select the first 600 indices for each value
        selected_indices_ones = indices_ones[:600]
        selected_indices_zero = indices_zero[:600]

        combined_indices = np.concatenate([selected_indices_ones, selected_indices_zero])
        np.random.shuffle(combined_indices)

        if len(combined_indices) == 1200:
            random_arrays = patients_array[i][combined_indices]
            random_events = eventsArray[i][combined_indices]
            dataArrayFiltered.append(random_arrays)
            eventsArrayFiltered.append(random_events)
    
    dataArrayFiltered = [item[:][:, :] for item in dataArrayFiltered]
    A = np.array(dataArrayFiltered)
    B = np.array(eventsArrayFiltered)
    C = A.reshape(-1, A.shape[-1])
    D = B.reshape(-1)

     # Calculate mean and standard deviation
    mean = np.mean(C)
    std = np.std(C)

    # Normalize the signals to zero-mean and unit variance
    C = (C - mean) / std

    # Wave filtering with Butterworth
    gamma_wave, beta_wave, theta_wave, alpha_wave, delta_wave = [],[],[],[],[]
    for i in range(0, len(C)):
        N = 128
        v = C[i]

        f = 1/sqrt(2)
        H = Hl(N,5,f)

        assert isclose(sum(H.T - linalg.inv(H)), 0), 'H is not orthogonal'

        w = dot(H,v)

        linf = N//2; lsup = N  ; D1 = w[linf:lsup]; wD1 = zeros_like(v); wD1[linf:lsup] = D1; #print(linf, lsup)
        linf //= 2 ; lsup //= 2; D2 = w[linf:lsup]; wD2 = zeros_like(v); wD2[linf:lsup] = D2; #print(linf, lsup)
        linf //= 2 ; lsup //= 2; D3 = w[linf:lsup]; wD3 = zeros_like(v); wD3[linf:lsup] = D3; #print(linf, lsup)
        linf //= 2 ; lsup //= 2; D4 = w[linf:lsup]; wD4 = zeros_like(v); wD4[linf:lsup] = D4; #print(linf, lsup)
        linf //= 2 ; lsup //= 2; D5 = w[linf:lsup]; wD5 = zeros_like(v); wD5[linf:lsup] = D5; #print(linf, lsup)
        lsup = linf; linf = 0  ; C5 = w[linf:lsup]; wC5 = zeros_like(v); wC5[linf:lsup] = C5; #print(linf, lsup)

        gamma_wave.append(dot(H.T, wD1))
        beta_wave.append(dot(H.T, wD2))
        alpha_wave.append(dot(H.T, wD3))
        theta_wave.append(dot(H.T, wD4))
        delta_wave.append(dot(H.T, wC5+wD5))
    gamma_wave = np.array(gamma_wave)
    beta_wave = np.array(beta_wave)
    alpha_wave = np.array(alpha_wave)
    theta_wave = np.array(theta_wave)
    delta_wave = np.array(delta_wave)

    # CNN Model creation
    input = tf.keras.Input(shape=(128,1))
    model1 = CNNCell1.CNNModel(input)

    combined = tf.keras.layers.Concatenate()([model1])
    output = tf.keras.layers.Dense(64, activation='relu')(combined) 
    output = tf.keras.layers.Dropout( 0.2, noise_shape=None, seed=None)(output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

    concatenated_model = tf.keras.Model(inputs=[input], outputs=output,name= 'Concatenated_Model_Frequency')
    concatenated_model.summary()

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

    concatenated_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    num_folds = 5
    kf = KFold(n_splits=num_folds)

    for fold, (train_indices, val_indices) in enumerate(kf.split(delta_wave)):
        print(f"Fold {fold + 1}/{num_folds}")

        delta_wave_train = delta_wave[train_indices]
        theta_wave_train = theta_wave[train_indices]
        alpha_wave_train = alpha_wave[train_indices]
        beta_wave_train = beta_wave[train_indices]
        gamma_wave_train = gamma_wave[train_indices]
        labels_train = D[train_indices]

        delta_wave_val = delta_wave[val_indices]
        theta_wave_val = theta_wave[val_indices]
        alpha_wave_val = alpha_wave[val_indices]
        beta_wave_val = beta_wave[val_indices]
        gamma_wave_val = gamma_wave[val_indices]
        labels_val = D[val_indices]

        concatenated_model.fit(
            x=[delta_wave_train],
            y=labels_train,
            batch_size=64,
            epochs=20,
            validation_data=([delta_wave_val], labels_val)
        )

    predictions = concatenated_model.predict([delta_wave_val])
    rounded_predictions = np.round(predictions)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels_val, rounded_predictions)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Alerta', 'Sonolência'], yticklabels=['Alerta', 'Sonolência'])
    plt.xlabel('Labels Previstos')
    plt.ylabel('Labels verdadeiros')
    plt.tight_layout()
    plt.show()

