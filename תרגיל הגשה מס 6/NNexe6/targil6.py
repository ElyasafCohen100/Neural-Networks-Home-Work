# ============================================================================================== #
#                                Exercise 6: Neural Networks                                     #
# ============================================================================================== #
#  Submitted by: Elyasaf Cohen 311557227 , Yakir Yohanan 312252034 and Pazit Akbashev 213527302  #
# ============================================================================================== #

import tensorflow as tf

from keras import Model
from keras.src.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten
from keras.src.legacy.preprocessing.image import ImageDataGenerator


# ============= Set the path to the dataset directory ============ #
dataset_dir = "/dataSet"

# =============================== Create an image generator with data augmentation ============================ #
# ============ Load and split the dataset into train/test sets ========== #
data_gen = ImageDataGenerator(rescale=0.255, validation_split=0.3)  # Normalize pixel values to [0, 1]

train_data = data_gen.flow_from_directory('dataSet',
                                          target_size=(224, 224),
                                          batch_size=32,
                                          class_mode='categorical',
                                          subset='training')

test_data = data_gen.flow_from_directory('dataSet',
                                         target_size=(224, 224),
                                         batch_size=32,
                                         class_mode='categorical',
                                         subset='validation')

# ===================== Define the CNN model without Transfer Learning and without Dropout ==================== #
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 categories
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ============= Train the model ============ #
cnn_model.fit(train_data, epochs=15, validation_data=test_data)

# ============ Evaluate the CNN model without Transfer Learning and without Dropout ========== #
# This step is already included in the 'fit' method with the 'validation_data' parameter

# ============ Add data augmentation to the training process ========== #
augmented_data_gen = ImageDataGenerator(
    rescale=0.255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
augmented_train_data = augmented_data_gen.flow_from_directory('dataSet',
                                                              target_size=(224, 224),
                                                              batch_size=32,
                                                              class_mode='categorical',
                                                              subset='training')

augmented_test_data = augmented_data_gen.flow_from_directory('dataSet',
                                                             target_size=(224, 224),
                                                             batch_size=32,
                                                             class_mode='categorical',
                                                             subset='validation')

# ============= Train the model with augmented data ============ #
cnn_model.fit(augmented_train_data, epochs=15, validation_data=augmented_test_data)


# ============ Evaluate the CNN model with data augmentation ========== #
# This step is already included in the 'fit' method with the 'validation_data' parameter

# ============ Use Transfer Learning with VGG16 ========== #
base_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
for layer in base_model.layers:
    layer.trainable = False  # Freeze the convolutional base

x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)  # Add dropout for regularization
x = Dense(4, activation='softmax')(x)  # 4 categories

transfer_model = Model(base_model.input, x)

transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ============= Train the transfer learning model ============ #
transfer_model.fit(augmented_train_data, epochs=15, validation_data=augmented_test_data)

# ============ Evaluate the transfer learning model ========== #
# This step is already included in the 'fit' method with the 'validation_data' parameter
