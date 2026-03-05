# ================================================================== #
#                   Exercise 4.2: Neural Networks                    #
# ================================================================== #
# Submitted by: Elyasaf Cohen 311557227 and Yakir Yohanan 312252034  #
# ================================================================== #

# ============================ Building a CNN using Keras' Sequential capabilities ============================ #

# Import necessary libraries
from __future__ import print_function
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Display shape of an image
print(x_train[444].shape)

# Convert class vectors to binary class matrices
num_classes = 10
print('Original class label:', y_train[442])
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print('Modified class label:', y_train[442])

# Normalize data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Build the CNN model with padding in all layers
model_3 = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),  # Input layer: 32x32 RGB image
        keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation="relu"),  # 1st Conv layer with 32 filters
        keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation="relu"),  # 2nd Conv layer with 64 filters
        keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),  # Max pooling layer with padding

        keras.layers.Flatten(),  # Flatten layer to transition to fully connected layers
        keras.layers.Dense((512), activation="relu"),  # Fully connected layer with 512 neurons
        keras.layers.Dropout(0.5),  # Dropout layer with 50% dropout rate
        keras.layers.Dense(num_classes, activation="softmax"),  # Output layer with softmax activation for classification
    ]
)

# Display model summary
model_3.summary()

# Compile the model
model_3.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
batch_size = 32
num_epochs = 10
model_3.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_data=(x_test, y_test),
            shuffle=True)

# Evaluate the trained model
score = model_3.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
