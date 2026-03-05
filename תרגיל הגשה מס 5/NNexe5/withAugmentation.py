# ================================================================== #
#                   Exercise 5: Neural Networks                      #
# ================================================================== #
# Submitted by: Elyasaf Cohen 311557227 and Yakir Yohanan 312252034  #
# ================================================================== #

# ======================================== CNN With Augmentation ===================================== #

# ============ Import necessary libraries ============ #
import os
import zipfile
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt  # Add this import

# ============ Set TensorFlow environment variable ============ #
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
train_generator_aug = train_datagen_aug.flow_from_directory(
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

# ============ Model setup ============ #
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# ============ Compile the model ============ #
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# ============ Training with Augmentation ============ #
history_aug = model.fit(
    train_generator_aug,
    epochs=15,
    validation_data=validation_generator,
    verbose=2
)

# ============ Plotting results for accuracy ============ #
plt.plot(history_aug.history['acc'], label='Training accuracy (Augmentation)')
plt.plot(history_aug.history['val_acc'], label='Validation accuracy (Augmentation)')
plt.title('Training and validation accuracy with Augmentation')
plt.legend()
plt.show()

# ============ Plotting results for loss ============ #
plt.plot(history_aug.history['loss'], label='Training loss (Augmentation)')
plt.plot(history_aug.history['val_loss'], label='Validation loss (Augmentation)')
plt.title('Training and validation loss with Augmentation')
plt.legend()
plt.show()
