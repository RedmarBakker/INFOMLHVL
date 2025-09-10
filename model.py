from tensorflow import keras
import numpy as np

def create_simple_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(256, input_shape=(784,)))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

    return model

if __name__ == "__main__":
    create_simple_model()
