# ================================================================== #
#                   Exercise 5: Neural Networks                      #
# ================================================================== #
# Submitted by: Elyasaf Cohen 311557227 and Yakir Yohanan 312252034  #
# ================================================================== #

# ================================== CNN With Augmentation And DropOut ================================= #

# ============ Import necessary libraries ============ #
import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from keras.src.legacy.preprocessing.image import ImageDataGenerator


# ============ Define path to the zip file with cat and dog images ============ #
local_zip = "C:\\Users\\uriel\\OneDrive\\Desktop\\cats_and_dogs_filtered.zip"
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

# ============ Define base directory for images ============ #
base_dir = '/tmp/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# ============ Define directories for training and validation images ============ #
train_cats_dir = os.path.join(train_dir, 'cats')  # Directory with training cat images
train_dogs_dir = os.path.join(train_dir, 'dogs')  # Directory with training dog images
validation_cats_dir = os.path.join(validation_dir, 'cats')  # Directory with validation cat images
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # Directory with validation dog images

# ============ List the filenames of training images ============ #
train_cat_fnames = os.listdir(train_cats_dir)
print(train_cat_fnames[:10])

train_dog_fnames = os.listdir(train_dogs_dir)
train_dog_fnames.sort()
print(train_dog_fnames[:10])

# ============ Print the number of images in each directory ============ #
print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))

# ============ Parameters for plotting images; we'll display images in a 4x4 grid ============ #
nrows = 4
ncols = 4

# ============ Index for iterating over images ============ #
pic_index = 0

# ============ Set up matplotlib fig and size it to fit 4x4 images ============ #
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_cat_pix = [os.path.join(train_cats_dir, fname) for fname in train_cat_fnames[pic_index - 8:pic_index]]
next_dog_pix = [os.path.join(train_dogs_dir, fname) for fname in train_dog_fnames[pic_index - 8:pic_index]]

for i, img_path in enumerate(next_cat_pix + next_dog_pix):
    sp = plt.subplot(nrows, ncols, i + 1)  # Set up subplot; subplot indices start at 1
    sp.axis('Off')  # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

# ============ Build a small ConvNet model for 72% accuracy ============ #
# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for the color channels: R, G, and B
model = keras.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(16, 3, activation='relu'),  # First convolution extracts 16 filters of size 3x3
    layers.MaxPooling2D(2),  # MaxPooling layer with a 2x2 window
    layers.Conv2D(32, 3, activation='relu'),  # Second convolution extracts 32 filters of size 3x3
    layers.MaxPooling2D(2),  # MaxPooling layer with a 2x2 window
    layers.Conv2D(64, 3, activation='relu'),  # Third convolution extracts 64 filters of size 3x3
    layers.MaxPooling2D(2),  # MaxPooling layer with a 2x2 window
    layers.Flatten(),  # Flatten the feature map to a 1D vector
    layers.Dropout(0.5),  # Add Dropout
    layers.Dense(512, activation='relu'),  # Fully connected layer with 512 units
    layers.Dense(1, activation='sigmoid')  # Output layer with a single unit and sigmoid activation
])

model.summary()

# ============ Compile the model ============ #
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['acc'])

# ============ Scale all images by 1./255 ============ #
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# ============ Flow training images in batches of 20 using train_datagen generator ============ #
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Source directory for training images
    target_size=(150, 150),  # Resize all images to 150x150
    batch_size=20,
    class_mode='binary')  # Because we use binary_crossentropy loss, we need binary labels

# ============ Flow validation images in batches of 20 using val_datagen generator ============ #
validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

# ============ Train the model ============ #
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    verbose=2)

# ============ Retrieve a list of accuracy results on training and validation sets for each training epoch ============ #
acc = history.history['acc']
val_acc = history.history['val_acc']

# ============ Retrieve a list of loss results on training and validation sets for each training epoch ============ #
loss = history.history['loss']
val_loss = history.history['val_loss']

# ============ Get number of epochs ============ #
epochs = range(len(acc))

# ============ Plot training and validation accuracy per epoch ============ #
plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

# ============ Plot training and validation loss per epoch ============ #
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()
