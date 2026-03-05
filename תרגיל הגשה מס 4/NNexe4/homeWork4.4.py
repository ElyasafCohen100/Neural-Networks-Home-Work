# ================================================================== #
#                   Exercise 4.4: Neural Networks                      #
# ================================================================== #
# Submitted by: Elyasaf Cohen 311557227 and Yakir Yohanan 312252034  #
# ================================================================== #

# ============================ Building a CNN to classify images in the CIFAR-10 Dataset ============================ #


# ======================== Building Convolutional Neural Nets ======================== #
from __future__ import print_function

import keras
import inline
import matplotlib
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

# =================== The data, shuffled and split between train and test sets: =================== #
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# ======== Each image is a 32 x 32 x 3 numpy array ======== #
# Notice that this time the pictures have color (RGB color with 3 channels).
x_train[444].shape


# On my local machine I had to write:
# from keras.utils.np_utils import to_categorical'''
# from keras.src.utils.np_utils import to_categorical

num_classes = 10
print(y_train[442])

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train[442])


# ========= As before, let's make everything float and scale ========= #
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# ======== Let's build a CNN using Keras' Sequential capabilities ======== #

model_1 = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation="relu"),
        keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense((512), activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model_1.summary()

batch_size = 32
num_epochs = 10

# =========== initiate Adam optimizer like in the first example =========== #
model_1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# initiate RMSprop optimizer
# opt = keras.optimizers.RMSprop(lr=0.0005, decay=0.00001)

# Let's train the model using RMSprop
# model_1.compile(loss='categorical_crossentropy',
#              optimizer=opt,
#              metrics=['accuracy'])

model_1.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_data=(x_test, y_test),
            shuffle=True)

# ========= Evaluate the trained model ========= #
score = model_1.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


# ======================================== Task D ======================================== #

# Build model_5 based on the suggested architecture

model_5 = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation="relu"),
        keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation="relu"),
        keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model_5.summary()

# Compile model_5
model_5.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model_5
model_5.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_data=(x_test, y_test),
            shuffle=True)

# Evaluate model_5
score = model_5.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
