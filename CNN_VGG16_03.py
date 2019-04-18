# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:25:07 2019

@author: w10007346
"""

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

from keras import models
from keras import layers


num_classes = 4
image_size = 400

vgg=VGG16(include_top=False, pooling='avg', weights='imagenet',input_shape=(image_size, image_size, 3))
vgg.summary()


# Freeze the layers except the last 2 layers
for layer in vgg.layers[:-5]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg.layers:
    print(layer, layer.trainable)
    

# Create the model
my_model = models.Sequential()


# Add the vgg convolutional base model
my_model.add(vgg)
 
# Add new layers
my_model.add(layers.Dense(128, activation='relu'))
my_model.add(layers.BatchNormalization())
my_model.add(layers.Dropout(.3))
my_model.add(layers.Dense(num_classes, activation='softmax'))
# Show a summary of the model. Check the number of trainable parameters
my_model.summary()


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


