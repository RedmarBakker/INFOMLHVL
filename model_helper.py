from tensorflow import keras
import numpy as np

from ConvLayer import ConvLayer
from Model import Model


def create_simple_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(256, input_shape=(784,)))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

    return model

def create_conv_model():
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(learning_rate=float(1)),
                  metrics=['accuracy'])

    return model

def create_custom_conv_model():
    model = Model(input_shape=input.shape)

    model.add(ConvLayer())

    return model

def relu(feature_map):
    return np.maximum(feature_map, 0)

if __name__ == "__main__":
    create_simple_model()