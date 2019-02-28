# Nate Jermain
# Convolutional Neural Network for age estimation Menhaden Scales
# 2/6/19


from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_classes = 6
resnet_weights_path = 'C:/Users/w10007346/Dropbox/CNN/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True, vertical_flip=True)


train_generator = data_generator.flow_from_directory(
        'C:/Users/w10007346/Dropbox/CNN/Aggregate test images/train',
        target_size=(image_size, image_size),
        batch_size=8,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        'C:/Users/w10007346/Dropbox/CNN/Aggregate test images/valid',
        target_size=(image_size, image_size),
        class_mode='categorical')

my_new_model.fit_generator(
        train_generator,
        epochs=30,
        steps_per_epoch=398,
        validation_data=validation_generator,
        validation_steps=171)

print("finished")
