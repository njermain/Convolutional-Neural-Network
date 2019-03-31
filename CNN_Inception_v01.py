# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:25:07 2019

@author: w10007346
"""
import numpy as np
import pandas as pd
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.utils import class_weight

num_classes = 6

my_new_model = Sequential()
my_new_model.add(InceptionV3(include_top=False, pooling='avg', weights='imagenet'))
my_new_model.add(Dense(num_classes, activation='softmax'))

my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

from tensorflow.python.keras.applications.inception_v3 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 299
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True, vertical_flip=True, rotation_range=90)


train_generator = data_generator.flow_from_directory(
        'C:/Users/w10007346/Dropbox/CNN/Aggregate test images/train',
        target_size=(image_size, image_size),
        batch_size=100,
        class_mode='categorical')


validation_generator = data_generator.flow_from_directory(
        'C:/Users/w10007346/Dropbox/CNN/Aggregate test images/valid',
        target_size=(image_size, image_size),
        batch_size=8,
        class_mode='categorical')
type(validation_generator)

# set class weigths given the unbalanced data set
class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(train_generator.classes), 
            train_generator.classes)

my_new_model.fit_generator(
        train_generator,
        epochs=10,
        steps_per_epoch=32,
        validation_data=validation_generator,
        validation_steps=171, class_weight=class_weights)

# lets see the predictions on the validation set
validation_generator.reset()
pred=my_new_model.predict_generator(validation_generator,verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
predictions.count('2')

filenames=validation_generator.filenames
results=pd.DataFrame({"Filename":filenames,"Predictions":predictions})
results.to_csv("results.csv",index=False)


