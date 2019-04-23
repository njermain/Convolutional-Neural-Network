# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:31:54 2019

@author: w10007346
"""
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


from keras import models

from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.models import Model


num_classes = 4
image_size = 400

vgg=VGG16(include_top=False, pooling='avg', weights='imagenet',input_shape=(image_size, image_size, 3))
vgg.summary()


layer_name = 'input_3'
sct_model= Model(inputs=vgg.input, outputs=vgg.get_layer(layer_name).output)

sct_model.summary()

  

# Create the model
my_model = models.Sequential()


# Add the vgg convolutional base model
my_model.add(sct_model)
 
# Add new layers
my_model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
my_model.add(MaxPooling2D((2, 2), padding='same'))
my_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
my_model.add(MaxPooling2D((2, 2), padding='same'))
my_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
my_model.add(MaxPooling2D((2, 2), padding='same'))
my_model.add(GlobalAveragePooling2D())
my_model.add(Dense(128, activation='relu'))
my_model.add(BatchNormalization())
my_model.add(Dense(num_classes, activation='softmax'))


# Show a summary of the model. Check the number of trainable parameters
my_model.summary()

# compile model 
my_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = data_generator.flow_from_directory(
        'C:/Users/w10007346/Dropbox/CNN/FullAugment/train',
        target_size=(image_size, image_size),
        batch_size=12,
        class_mode='categorical')


validation_generator = data_generator.flow_from_directory(
        'C:/Users/w10007346/Dropbox/CNN/FullAugment/valid',
        target_size=(image_size, image_size),
        batch_size=12,
        class_mode='categorical')


my_model.fit_generator(
        train_generator,
        epochs=20,
        steps_per_epoch=283,
        validation_data=validation_generator,
        validation_steps=91)