from tensorflow import keras
import numpy as np

def create_simple_data(x_train, y_train, x_test, y_test):
    # Flatten
    x_train_reshape = x_train.reshape(x_train.shape[0], -1)
    x_test_reshape = x_test.reshape(x_test.shape[0], -1)

    # Normalize data
    x_train_norm = x_train_reshape.astype("float32") / 255.0
    x_test_norm = x_test_reshape.astype("float32") / 255.0

    # hot one encode the labels
    y_train_hot = keras.utils.to_categorical(y_train, 10)
    y_test_hot = keras.utils.to_categorical(y_test, 10)

    return x_train_norm, y_train_hot, x_test_norm, y_test_hot

def create_conv_data(x_train, y_train, x_test, y_test):
    # Add 1 new dimension holding RGB (only 1 channel)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Normalize data
    x_train_norm = x_train.astype("float32") / 255.0
    x_test_norm = x_test.astype("float32") / 255.0

    # hot one encode the labels
    y_train_hot = keras.utils.to_categorical(y_train, 10)
    y_test_hot = keras.utils.to_categorical(y_test, 10)

    return x_train_norm, y_train_hot, x_test_norm, y_test_hot
