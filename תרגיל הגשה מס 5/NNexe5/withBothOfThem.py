# ================================================================== #
#                   Exercise 5: Neural Networks                      #
# ================================================================== #
# Submitted by: Elyasaf Cohen 311557227 and Yakir Yohanan 312252034  #
# ================================================================== #

# ================================== CNN With Augmentation And DropOut ================================= #

# ============ Import necessary libraries ============ #
import os
import zipfile
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt  # Add this import

# ============ Define path to the zip file with cat and dog images ============ #
local_zip = "C:\\Users\\uriel\\OneDrive\\Desktop\\cats_and_dogs_filtered.zip"

# ============ Extract the zip file to a temporary directory ============ #
extraction_path = "C:\\Users\\uriel\\OneDrive\\Desktop\\cats_and_dogs_filtered"
with zipfile.ZipFile(local_zip, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

# ============ Define base directory for images ============ #
base_dir = os.path.join(extraction_path, 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# ============ Augmentation with ImageDataGenerator ============ #
train_datagen_aug = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# ============ Create data generators for training and validation sets ============ #
train_generator_aug_dropout = train_datagen_aug.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# ============ Model setup with Dropout ============ #
model_aug_dropout = models.Sequential()
model_aug_dropout.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model_aug_dropout.add(layers.MaxPooling2D((2, 2)))
model_aug_dropout.add(layers.Dropout(0.5))
model_aug_dropout.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_aug_dropout.add(layers.MaxPooling2D((2, 2)))
model_aug_dropout.add(layers.Dropout(0.5))
model_aug_dropout.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_aug_dropout.add(layers.MaxPooling2D((2, 2)))
model_aug_dropout.add(layers.Dropout(0.5))
model_aug_dropout.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_aug_dropout.add(layers.MaxPooling2D((2, 2)))
model_aug_dropout.add(layers.Flatten())
model_aug_dropout.add(layers.Dropout(0.5))
model_aug_dropout.add(layers.Dense(512, activation='relu'))
model_aug_dropout.add(layers.Dropout(0.5))
model_aug_dropout.add(layers.Dense(1, activation='sigmoid'))

# ============ Compile the model ============ #
model_aug_dropout.compile(loss='binary_crossentropy',
                          optimizer='rmsprop',
                          metrics=['acc'])

# ============ Training with Augmentation and Dropout ============ #
history_aug_dropout = model_aug_dropout.fit(
    train_generator_aug_dropout,
    epochs=15,
    validation_data=validation_generator,
    verbose=2
)

# ============ Plotting results for accuracy ============ #
plt.plot(history_aug_dropout.history['acc'], label='Training accuracy (Aug + Dropout)')
plt.plot(history_aug_dropout.history['val_acc'], label='Validation accuracy (Aug + Dropout)')
plt.title('Training and validation accuracy with Augmentation and Dropout')
plt.legend()
plt.show()

# ============ Plotting results for loss ============ #
plt.plot(history_aug_dropout.history['loss'], label='Training loss (Aug + Dropout)')
plt.plot(history_aug_dropout.history['val_loss'], label='Validation loss (Aug + Dropout)')
plt.title('Training and validation loss with Augmentation and Dropout')
plt.legend()
plt.show()
