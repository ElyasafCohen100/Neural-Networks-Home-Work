# ================================================================== #
#                   Exercise 4.3: Neural Networks                    #
# ================================================================== #
# Submitted by: Elyasaf Cohen 311557227 and Yakir Yohanan 312252034  #
# ================================================================== #
'''
Build model_4 using the MNIST network architecture with padding.
'''

# ================ Description: A simple convent that achieves ~99% test accuracy on MNIST. ================ #
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense

# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= STAGE 1 *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= #
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_index = 100  # You may select anything up to 60,000
print(y_train[image_index])  # The label is 5
plt.imshow(x_train[image_index], cmap='Greys')

image_index = 101  # You may select anything up to 60,000
print(y_train[image_index])  # The label is 3
plt.imshow(x_train[image_index], cmap='Greys')

# Normalize and reshape the data
num_classes = 10  # number of categories in MNIST
input_shape = (28, 28, 1)  # the MNIST image's dim

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= STAGE 2 *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= #
# One-hot encode the labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= STAGE 3 *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= #
# Build the model
model_4 = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model_4.summary()

# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= STAGE 4 *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= #
# Compile and train the model
model_4.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

batch_size = 512
num_epochs = 10

model_4.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_data=(x_test, y_test),
            shuffle=True)

# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= STAGE 5 *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= #
# Evaluate the model
score = model_4.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
