import numpy as np
import tensoflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# object for augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# loading training dataset
training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target=(64,64),
    batch_size=32,
    class_mode='binary'
)

# object for augmentation for test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory()