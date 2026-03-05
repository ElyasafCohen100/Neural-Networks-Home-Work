# ================================================================== #
#                   Exercise 4.1: Neural Networks                    #
# ================================================================== #
# Submitted by: Elyasaf Cohen 311557227 and Yakir Yohanan 312252034  #
# ================================================================== #


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
import time

# =================== The data, shuffled and split between train and test sets: =================== #
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# ======== Each image is a 32 x 32 x 3 numpy array ======== #
# Notice that this time the pictures have color (RGB color with 3 channels).
x_train[444].shape

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


# ======== Function to create the CNN model ======== #
def create_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),  # שכבת הקלט: תמונה בגודל 32x32 עם 3 ערוצים (RGB)
            keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation="relu"),
            # שכבת קונבולוציה ראשונה עם 32 פילטרים
            keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation="relu"),
            # שכבת קונבולוציה שנייה עם 64 פילטרים
            keras.layers.MaxPooling2D(pool_size=(2, 2)),  # שכבת מקסימום פולינג להפחתת מימדי התמונה
            keras.layers.Flatten(),  # שטוח את המטריצה למערך אחד
            keras.layers.Dense((512), activation="relu"),  # שכבת Fully Connected עם 512 נוירונים
            keras.layers.Dropout(0.5),  # שכבת Dropout עם 50% הסרה אקראית של נוירונים
            keras.layers.Dense(num_classes, activation="softmax"),  # שכבת פלט עם 10 נוירונים ותפקוד softmax לסיווג
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# ========= Training models with different batch sizes ========= #
batch_sizes = [4, 128, 1024]
models = [create_model() for _ in batch_sizes]

results = []

for batch_size, model in zip(batch_sizes, models):
    print(f"Training model with batch size: {batch_size}")
    start_time = time.time()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=10, validation_data=(x_test, y_test),
                        shuffle=True)
    end_time = time.time()

    elapsed_time = end_time - start_time
    score = model.evaluate(x_test, y_test, verbose=0)
    accuracy = score[1]

    results.append((batch_size, accuracy, elapsed_time))

# הדפסת התוצאות
for batch_size, accuracy, elapsed_time in results:
    print(f"Batch size: {batch_size}, Accuracy: {accuracy}, Elapsed time: {elapsed_time}")

# המודל הטוב ביותר והמהיר ביותר
best_model = max(results, key=lambda x: x[1])
fastest_model = min(results, key=lambda x: x[2])

print(f"Best model: Batch size: {best_model[0]}, Accuracy: {best_model[1]}")
print(f"Fastest model: Batch size: {fastest_model[0]}, Elapsed time: {fastest_model[2]}")
