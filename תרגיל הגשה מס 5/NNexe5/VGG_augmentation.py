# ================================================================== #
#                   Exercise 5: Neural Networks                      #
# ================================================================== #
# Submitted by: Elyasaf Cohen 311557227 and Yakir Yohanan 312252034  #
# ================================================================== #

# =================================== CNN With VGG Augmentation =================================== #

# ============ Import necessary libraries ============ #
import os
import zipfile
import matplotlib.pyplot as plt

# ============ Import TensorFlow and Keras libraries for building the neural network ============ #
from tensorflow.keras import layers, models
from keras.src.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import RMSprop
from keras.src.legacy.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============ Set TensorFlow environment variable ============ #
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ============ Define paths to the zip file and extraction directory ============ #
local_zip = "C:\\Users\\uriel\\OneDrive\\Desktop\\cats_and_dogs_filtered.zip"
extraction_path = "C:\\Users\\uriel\\OneDrive\\Desktop\\cats_and_dogs_filtered"

# ============ Extract the zip file ============ #
with zipfile.ZipFile(local_zip, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

# ============ Define paths to the training and validation directories ============ #
base_dir = os.path.join(extraction_path, 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# ============ Create ImageDataGenerators for training and validation sets with augmentation ============ #
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

# ============ Load the VGG16 model without the top fully connected layers ============ #
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# ============ Freeze the convolutional base layers ============ #
conv_base.trainable = False

# ============ Build the new model ============ #
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# ============ Compile the model ============ #
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['acc'])

# ============ Train the model with the training data and validate with the validation data ============ #
history_aug = model.fit(
    train_generator_aug,
    epochs=15,
    validation_data=validation_generator,
    verbose=2
)

# ============ Plot the training and validation accuracy ============ #
plt.plot(history_aug.history['acc'], label='Training accuracy (Augmentation)')
plt.plot(history_aug.history['val_acc'], label='Validation accuracy (Augmentation)')
plt.title('Training and validation accuracy with Augmentation')
plt.legend()
plt.show()

# ============ Plot the training and validation loss ============ #
plt.plot(history_aug.history['loss'], label='Training loss (Augmentation)')
plt.plot(history_aug.history['val_loss'], label='Validation loss (Augmentation)')
plt.title('Training and validation loss with Augmentation')
plt.legend()
plt.show()
