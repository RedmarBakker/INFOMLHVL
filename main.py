from tensorflow import keras
import numpy as np
# import matplotlib
#
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import ssl
import certifi
from model_helper import *
from Model import Model

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# Get data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

model_type = 'custom'

if model_type == 'simpel':
    x_train, y_train, x_test, y_test = create_simple_data(x_train, y_train, x_test, y_test)

    # Create model and fit
    model = create_simple_model()
    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=12,
                        verbose=1,
                        validation_split=0.2)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
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
elif model_type == 'conv':
    x_train, y_train, x_test, y_test = create_conv_data(x_train, y_train, x_test, y_test)

    model = create_conv_model()

    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=6,
                        verbose=1,
                        validation_split=0.2)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Loss: {loss}, Accuracy: {accuracy}")

    # Extract metrics from history
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    epochs = range(1, len(loss) + 1)

    # Plot loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, "bo-", label="Training loss")
    plt.plot(epochs, val_loss, "ro-", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, acc, "bo-", label="Training accuracy")
    plt.plot(epochs, val_acc, "ro-", label="Validation accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

elif model_type == 'custom':
    # Input
    sample_image = x_train[0]
    sample_label = y_train[0]

    # Define kernels
    horizontal_kernel = np.array(
        [[-1, -1, -1],
         [2, 2, 2],
         [-1, -1, -1]]
    )

    vertical_kernel = np.array(
        [[-1, 2, -1],
         [-1, 2, -1],
         [-1, 2, -1]]
    )

    model = create_custom_conv_model([horizontal_kernel])
    output = model.process(sample_image)

    # Plot the results
    plt.imshow(output, cmap="gray")  # grayscale colormap
    plt.title(f"Label: {sample_label}")
    plt.axis("off")  # hide axes
    plt.show()
    #
    # plt.imshow(vertical_output, cmap="gray")  # grayscale colormap
    # plt.title(f"Label: {sample_label}")
    # plt.axis("off")  # hide axes
    # plt.show()

    # plt.imshow(hor_ver_output, cmap="gray")  # grayscale colormap
    # plt.title(f"Label: {sample_label}")
    # plt.axis("off")  # hide axes
    # plt.show()
    #
    # plt.imshow(ver_hor_output, cmap="gray")  # grayscale colormap
    # plt.title(f"Label: {sample_label}")
    # plt.axis("off")  # hide axes
    # plt.show()