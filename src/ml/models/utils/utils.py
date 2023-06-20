import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf

def load_dataset():
    # Load the letters dataset
    letters_df = pd.read_csv(Path(__file__).parent / 'A_Z Handwritten Data.csv')

    # Reshape the images to be 28x28 pixels and convert the pixel values to 0-1
    letters_X = letters_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
    letters_y = letters_df.iloc[:, 0].values

    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (mnist_X_train, mnist_y_train), (mnist_X_test, mnist_y_test) = mnist.load_data()

    # Reshape the MNIST images to be 28x28 pixels and convert the pixel values to 0-1
    mnist_X_train = mnist_X_train.reshape(-1, 28, 28, 1) / 255.0
    mnist_X_test = mnist_X_test.reshape(-1, 28, 28, 1) / 255.0

    # Combine the datasets
    X_train = np.concatenate((mnist_X_train, letters_X[:20000]))
    y_train = np.concatenate((mnist_y_train, letters_y[:20000]))
    X_test = np.concatenate((mnist_X_test, letters_X[20000:]))
    y_test = np.concatenate((mnist_y_test, letters_y[20000:]))

    # Shuffle the datasets
    idx_train = np.random.permutation(len(X_train))
    idx_test = np.random.permutation(len(X_test))
    X_train, y_train = X_train[idx_train], y_train[idx_train]
    X_test, y_test = X_test[idx_test], y_test[idx_test]
    return X_train, y_train, X_test, y_test