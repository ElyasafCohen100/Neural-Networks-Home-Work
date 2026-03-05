# ================================================================== #
#                   Exercise 2: Neural Networks                      #
# ================================================================== #
# Submitted by: Elyasaf Cohen 311557227 and Yakir Yohanan 312252034  #
# ================================================================== #
'''
Simple MNIST convent (convolution network)
Author: f-chollet (https://keras.io/examples/vision/mnist_convnet/)
*******************************
* Date created: 2015/06/19    *
* Last modified: 2020/04/21   *
*******************************
Parts also taken from:
https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
'''

# ================ Description: A simple convent that achieves ~99% test accuracy on MNIST. ================ #
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense

# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= STAGE 1 *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= #
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_index = 100  # You may select anything up to 60,000
print(y_train[image_index])  # The label is 5
plt.imshow(x_train[image_index], cmap='Greys')

image_index = 101  # You may select anything up to 60,000
print(y_train[image_index])  # The label is 3
plt.imshow(x_train[image_index], cmap='Greys')


# Now, let's do the data preparation:
# As greyscale is between 0-255,
# we need to normalize the values (like we've done before).
# We also need to manually tell Keras that we have one channel (greyscale).

num_classes = 10  # number of categories in MNIST
input_shape = (28, 28, 1)  # the MNIST image's dim

# ============ Scale images to [0, 1] range ============= #
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
print(x_train.shape)  # shape before

# ======== Make sure images have shape (28, 28, 1) ======= #
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# ============ shape after ============ #
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= STAGE 2 *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= #

# ======================== Now let's add one-hot encoding: ======================== #
print(y_train[101])  # print the label of the example number 101 in y_train

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train[101])

# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= STAGE 3 *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= #

# ========================== Build the model parameters: =========================== #
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= STAGE 4 *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= #

# =============================== Train the model: ================================== #
batch_size = 512
num_epochs = 10

# opt = keras.optimizers.RMSprop(lr=0.0005, decay=0.00001)
# model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=num_epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= STAGE 5 *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= #

# ============================ Evaluate the trained model ============================ #
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
