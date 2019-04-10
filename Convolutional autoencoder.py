# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 17:34:31 2019

Nate Jermain
Convolutional autoencoder to identify features 
"""




from tensorflow.python.keras.applications.inception_v3 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
image_size = 800
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True)

train_generator = data_generator.flow_from_directory(
        'C:/Users/w10007346/Dropbox/CNN/FewerClasses/train',
        target_size=(image_size, image_size),
        batch_size=8,
        class_mode='input')


validation_generator = data_generator.flow_from_directory(
        'C:/Users/w10007346/Dropbox/CNN/FewerClasses/valid',
        target_size=(image_size, image_size),
        batch_size=8,
        class_mode='input')



from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(800, 800, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(train_generator, train_generator,
                epochs=30,
                batch_size=128,
                shuffle=True,
                validation_data=(validation_generator, validation_generator))