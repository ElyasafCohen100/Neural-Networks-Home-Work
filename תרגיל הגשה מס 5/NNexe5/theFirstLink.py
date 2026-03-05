# ================================================================== #
#                   Exercise 5: Neural Networks                      #
# ================================================================== #
# Submitted by: Elyasaf Cohen 311557227 and Yakir Yohanan 312252034  #
# ================================================================== #

# ==========================================             =========================================== #

import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers

from tensorflow.keras.optimizers import RMSprop
from keras.src.legacy.preprocessing.image import ImageDataGenerator
# import tensorflow.keras.preprocessing.image


# נתיב לקובץ ZIP עם התמונות של חתולים וכלבים
local_zip = "C:\\Users\\uriel\\OneDrive\\Desktop\\cats_and_dogs_filtered.zip"
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

# ספריית הבסיס לתמונות
base_dir = '/tmp/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# ספריה עם תמונות אימון של חתולים
train_cats_dir = os.path.join(train_dir, 'cats')

# ספריה עם תמונות אימון של כלבים
train_dogs_dir = os.path.join(train_dir, 'dogs')

# ספריה עם תמונות ולידציה של חתולים
validation_cats_dir = os.path.join(validation_dir, 'cats')

# ספריה עם תמונות ולידציה של כלבים
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# רשימת שמות קבצי התמונות של חתולים באימון
train_cat_fnames = os.listdir(train_cats_dir)
print(train_cat_fnames[:10])

# רשימת שמות קבצי התמונות של כלבים באימון
train_dog_fnames = os.listdir(train_dogs_dir)
train_dog_fnames.sort()
print(train_dog_fnames[:10])

# הדפסת מספר התמונות בכל ספריה
print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))

# פרמטרים לגרף; נציג תמונות בתצורת 4x4
nrows = 4
ncols = 4

# אינדקס לחזרה על תמונות
pic_index = 0

# הגדרת matplotlib fig והתאמת הגודל
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_cat_pix = [os.path.join(train_cats_dir, fname)
                for fname in train_cat_fnames[pic_index - 8:pic_index]]
next_dog_pix = [os.path.join(train_dogs_dir, fname)
                for fname in train_dog_fnames[pic_index - 8:pic_index]]

for i, img_path in enumerate(next_cat_pix + next_dog_pix):
    # הגדרת subplot; אינדקסים של subplot מתחילים ב-1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')  # ביטול הצגת צירים (או גריד)

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

# בניית מודל קטן מ-ConvNet לקבלת דיוק של 72%
# מפת התכונות הקלט שלנו היא 150x150x3: 150x150 עבור פיקסלי התמונה ו-3 עבור ערוצי הצבע: R, G, ו-B
img_input = layers.Input(shape=(150, 150, 3))

model = keras.Sequential(
    [
        layers.Input(shape=(150, 150, 3)),
        # קונבולוציה ראשונה מחלצת 16 פילטרים בגודל 3x3
        # הקונבולוציה מלווה בשכבת מקסימום-פולינג עם חלון של 2x2
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(2),
        # קונבולוציה שנייה מחלצת 32 פילטרים בגודל 3x3
        # הקונבולוציה מלווה בשכבת מקסימום-פולינג עם חלון של 2x2
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(2),
        # קונבולוציה שלישית מחלצת 64 פילטרים בגודל 3x3
        # הקונבולוציה מלווה בשכבת מקסימום-פולינג עם חלון של 2x2
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        # שטוח את מפת התכונות למאגר חד-ממדי כדי שנוכל להוסיף שכבות מקושרות באופן מלא
        layers.Flatten(),
        # layers.Dropout(.4),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ]
)

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['acc'])

# כל התמונות יעברו סקיילינג ע"י 1./255
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# זרימת תמונות האימון במנות של 20 באמצעות train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,  # ספריית המקור לתמונות האימון
    target_size=(150, 150),  # כל התמונות ישנו גודל ל-150x150
    batch_size=20,
    # מכיוון שאנו משתמשים בהפסד binary_crossentropy, אנו זקוקים לתוויות בינאריות
    class_mode='binary')

# זרימת תמונות הוולידציה במנות של 20 באמצעות val_datagen generator
validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

# אימון המודל
history = model.fit(
    train_generator,
    # steps_per_epoch=100,  # 2000 תמונות = batch_size * steps
    epochs=10,
    validation_data=validation_generator,
    # validation_steps=50,  # 1000 תמונות = batch_size * steps
    verbose=2)

# הערכת דיוק ואובדן למודל

# אחזור רשימת תוצאות דיוק על סטי אימון וולידציה עבור כל תקופת אימון
acc = history.history['acc']
val_acc = history.history['val_acc']

# אחזור רשימת תוצאות אובדן על סטי אימון וולידציה עבור כל תקופת אימון
loss = history.history['loss']
val_loss = history.history['val_loss']

# קבלת מספר התקופות
epochs = range(len(acc))

# ציור גרף דיוק אימון וולידציה לפי תקופה
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('דיוק אימון וולידציה')

plt.figure()

# ציור גרף אובדן אימון וולידציה לפי תקופה
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('אובדן אימון וולידציה')
