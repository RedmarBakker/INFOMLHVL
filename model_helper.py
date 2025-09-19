from tensorflow import keras
import numpy as np

from AllLayers import *
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

def create_custom_conv_model(kernels, num_outputs):
    model = Model()

    model.add(ConvLayer().add_filters(kernels))
    model.add(ReLULayer())
    model.add(MaxPoolingLayer(pool_size=(2, 2)))
    model.add(NormalizeLayer())
    model.add(FCLayer(num_outputs))
    model.add(SoftmaxLayer())

    return model





if __name__ == "__main__":
    create_simple_model()