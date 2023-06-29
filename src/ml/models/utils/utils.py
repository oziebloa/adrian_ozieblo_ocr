from pathlib import Path

import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split


def load_data():
    (X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
    data = pd.read_csv(Path(__file__).parent / "A_Z Handwritten Data.csv")
    X = data.drop('0', axis=1)
    y = data['0']
    y += 10
    X_train_az, X_test_az, y_train_az, y_test_az = train_test_split(X, y, test_size=0.2)
    X_train_mnist, X_test_mnist, X_train_az, X_test_az = X_train_mnist / 255, X_test_mnist / 255, X_train_az / 255, X_test_az / 255
    X_train_mnist = X_train_mnist.reshape(-1, 28, 28, 1)
    X_test_mnist = X_test_mnist.reshape(-1, 28, 28, 1)
    X_train_az = X_train_az.values.reshape(-1, 28, 28, 1)
    X_test_az = X_test_az.values.reshape(-1, 28, 28, 1)
    X_train = np.concatenate((X_train_mnist, X_train_az), axis=0)
    y_train = np.concatenate((y_train_mnist, y_train_az), axis=0)
    X_test = np.concatenate((X_test_mnist, X_test_az), axis=0)
    y_test = np.concatenate((y_test_mnist, y_test_az), axis=0)
    y_train = to_categorical(y_train, 36)
    y_test = to_categorical(y_test, 36)

    return X_train, y_train, X_test, y_test
