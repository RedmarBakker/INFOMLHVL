from tensorflow import keras
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import ssl
import certifi
from model import create_simple_model

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# Get data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Flatten and normalize data
x_train_reshape = x_train.reshape(x_train.shape[0], -1)
x_test_reshape = x_test.reshape(x_test.shape[0], -1)

x_train_norm = x_train_reshape.astype("float32") / 255.0
x_test_norm = x_test_reshape.astype("float32") / 255.0

# hot one encode the labels
y_train_hot = keras.utils.to_categorical(y_train, 10)
y_test_hot = keras.utils.to_categorical(y_test, 10)

# Create model and fit
model = create_simple_model()
history = model.fit(x_train_norm, y_train_hot, batch_size=128,
epochs=12, verbose=1, validation_split=0.2)

loss, accuracy = model.evaluate(x_test_norm, y_test_hot, verbose=0)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# # Extract metrics from history
# loss = history.history["loss"]
# val_loss = history.history["val_loss"]
# acc = history.history["accuracy"]
# val_acc = history.history["val_accuracy"]
#
# epochs = range(1, len(loss) + 1)

# # Plot loss
# plt.figure(figsize=(8, 5))
# plt.plot(epochs, loss, "bo-", label="Training loss")
# plt.plot(epochs, val_loss, "ro-", label="Validation loss")
# plt.title("Training and Validation Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
#
# # Plot accuracy
# plt.figure(figsize=(8, 5))
# plt.plot(epochs, acc, "bo-", label="Training accuracy")
# plt.plot(epochs, val_acc, "ro-", label="Validation accuracy")
# plt.title("Training and Validation Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()



