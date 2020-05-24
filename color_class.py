#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.image as mpimg

local_zip = '/Users/kregan/School/ML/final/color-train.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/Users/kregan/School/ML/final/')
zip_ref.close()

local_zip = '/Users/kregan/School/ML/final/color-val.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/Users/kregan/School/ML/final/')
zip_ref.close()

local_zip = '/Users/kregan/School/ML/final/color-test.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/Users/kregan/School/ML/final/')
zip_ref.close()

black_dir = os.path.join('/Users/kregan/School/ML/final/color-train/color/black')
blue_dir = os.path.join('/Users/kregan/School/ML/final/color-train/color/blue')
brown_dir = os.path.join('/Users/kregan/School/ML/final/color-train/color/brown')
green_dir = os.path.join('/Users/kregan/School/ML/final/color-train/color/green')
orange_dir = os.path.join('/Users/kregan/School/ML/final/color-train/color/orange')
red_dir = os.path.join('/Users/kregan/School/ML/final/color-train/color/red')
violet_dir = os.path.join('/Users/kregan/School/ML/final/color-train/color/violet')
white_dir = os.path.join('/Users/kregan/School/ML/final/color-train/color/white')
yellow_dir = os.path.join('/Users/kregan/School/ML/final/color-train/color/yellow')
test_dir = os.path.join('/Users/kregan/School/ML/final/color-test/color')

black_files = os.listdir(black_dir)
blue_files = os.listdir(blue_dir)
brown_files = os.listdir(brown_dir)
green_files = os.listdir(green_dir)
orange_files = os.listdir(orange_dir)
red_files = os.listdir(red_dir)
violet_files = os.listdir(violet_dir)
white_files = os.listdir(white_dir)
yellow_files = os.listdir(yellow_dir)
test_files = os.listdir(test_dir)

if '.DS_Store' in black_files: black_files.remove('.DS_Store')
if '.DS_Store' in blue_files: blue_files.remove('.DS_Store')
if '.DS_Store' in brown_files: brown_files.remove('.DS_Store')
if '.DS_Store' in green_files: green_files.remove('.DS_Store')
if '.DS_Store' in orange_files: orange_files.remove('.DS_Store')
if '.DS_Store' in red_files: red_files.remove('.DS_Store')
if '.DS_Store' in violet_files: violet_files.remove('.DS_Store')
if '.DS_Store' in white_files: white_files.remove('.DS_Store')
if '.DS_Store' in yellow_files: yellow_files.remove('.DS_Store')
if '.DS_Store' in test_files: test_files.remove('.DS_Store')

print('total training images:', len(black_files+blue_files+brown_files+green_files+orange_files+red_files+
                                    violet_files+white_files+yellow_files))  

TRAINING_DIR = '/Users/kregan/School/ML/final/color-train/color'
training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150,150),
    class_mode='categorical'
)

VALIDATION_DIR = '/Users/kregan/School/ML/final/color-val/color'
validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150,150),
    class_mode='categorical'
)

TEST_DIR = '/Users/kregan/School/ML/final/color-test'
test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(150,150),
    class_mode='categorical',
    shuffle = False
)

model = tf.keras.models.Sequential([
    # Input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2), # factors to downscale by, (2,2) will halve
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),   # 2nd convo layer
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),  # 3rd convo layer
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),  # 4th convo layer
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),            # Flatten to DNN
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),      # hidden layer 
    tf.keras.layers.Dense(9, activation='softmax')      # 9 class 
])

model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(train_generator, epochs=22, validation_data = validation_generator, verbose = 1)
model.save("rps.h5")

pred = model.predict(test_generator)
pred_classes = pred.argmax(axis=-1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

style.use('ggplot')
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc=0)
plt.show()

test_files.sort()
for i in range(len(pred_classes)):
    img_path = '/Users/kregan/School/ML/final/color-test/color/' + test_files[i]
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()
    if pred_classes[i] == 0: print('black')
    if pred_classes[i] == 1: print('blue')
    if pred_classes[i] == 2: print('brown')
    if pred_classes[i] == 3: print('green')
    if pred_classes[i] == 4: print('orange')
    if pred_classes[i] == 5: print('red')
    if pred_classes[i] == 6: print('violet')
    if pred_classes[i] == 7: print('white')
    if pred_classes[i] == 8: print('yellow')

