# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:25:07 2019

@author: w10007346
"""

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

from keras import models
from keras import layers
from keras import optimizers

num_classes = 4
image_size = 800

vgg=VGG16(include_top=False, pooling='avg', weights='imagenet',input_shape=(image_size, image_size, 3))
vgg.summary()

#remove global_average_pooling2d_6 layer
from keras.models import Model
layer_name = 'block5_pool'
model2= Model(inputs=vgg.input, outputs=vgg.get_layer(layer_name).output)
model2.summary()

# Freeze the layers except the last 2 layers
for layer in model2.layers[:-1]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in model2.layers:
    print(layer, layer.trainable)
    



# Create the model
model = models.Sequential()


# Add the vgg convolutional base model
model.add(model2)
 
# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))
 
# Show a summary of the model. Check the number of trainable parameters
model.summary()


model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True)


train_generator = data_generator.flow_from_directory(
        'C:/Users/w10007346/Dropbox/CNN/FewerClasses/train',
        target_size=(image_size, image_size),
        batch_size=12,
        class_mode='categorical')


validation_generator = data_generator.flow_from_directory(
        'C:/Users/w10007346/Dropbox/CNN/FewerClasses/valid',
        target_size=(image_size, image_size),
        batch_size=12,
        class_mode='categorical')


# set class weigths given the unbalanced data set
#class_weights = class_weight.compute_class_weight(
#           'balanced',
#            np.unique(train_generator.classes), 
#            train_generator.classes)

model.fit_generator(
        train_generator,
        epochs=30,
        steps_per_epoch=283,
        validation_data=validation_generator,
        validation_steps=90)

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


