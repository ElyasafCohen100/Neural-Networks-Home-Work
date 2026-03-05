# ================================================================== #
#                   Exercise 5: Neural Networks                      #
# ================================================================== #
# Submitted by: Elyasaf Cohen 311557227 and Yakir Yohanan 312252034  #
# ================================================================== #

# ========================= Model with basic preprocessing and Visualization ========================= #

# ============ Import necessary libraries ============ #
import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ============ Import TensorFlow and Keras libraries for building the neural network ============ #
from tensorflow.keras import layers, models
from keras.src.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import RMSprop
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# ============ Define paths to the specific class directories ============ #
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# ============ List filenames in the training directories ============ #
train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

# ============ Print the total number of images in each directory ============ #
print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))

# ============ Display sample images from the dataset ============ #
nrows = 4
ncols = 4
pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_cat_pix = [os.path.join(train_cats_dir, fname)
                for fname in train_cat_fnames[pic_index - 8:pic_index]]
next_dog_pix = [os.path.join(train_dogs_dir, fname)
                for fname in train_dog_fnames[pic_index - 8:pic_index]]

for i, img_path in enumerate(next_cat_pix + next_dog_pix):
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

# ============ Create ImageDataGenerators for training and validation sets ============ #
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# ============ Create data generators for training and validation sets ============ #
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
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
model.add(layers.Dense(1, activation='sigmoid'))

# ============ Compile the model ============ #
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['acc'])

# ============ Train the model with the training data and validate with the validation data ============ #
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    verbose=2
)

# ============ Extract accuracy and loss data from the training history ============ #
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# ============ Plot the training and validation accuracy ============ #
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

# ============ Plot the training and validation loss ============ #
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
