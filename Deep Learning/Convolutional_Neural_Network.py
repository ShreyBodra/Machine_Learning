import numpy as np
import tensorflow as tf
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
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# object for augmentation for test data
test_datagen = ImageDataGenerator(rescale=1./255)

#loading test set
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)


# adding convolutional layers and applying max pooling
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    activation='relu',
    input_shape=[64, 64, 3]
))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# adding another layer
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    activation='relu'
))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# creating ann for classifying image
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# compiling model
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# training model
cnn.fit(training_set, test_set, epochs=25)



# testing model on single input
test_image = image.load_img('dataset/single_prediction/cat.4052.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
print(training_set.class_indices)
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'

# output
print(prediction)

