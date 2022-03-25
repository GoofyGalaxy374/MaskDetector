import os
from tkinter import HORIZONTAL
from unicodedata import category
from matplotlib.cbook import maxdict
from matplotlib.pyplot import colormaps
from numpy import reshape

from tensorflow.core.framework.dataset_options_pb2 import OptimizationOptions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import losses_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import pandas as pd
import cv2


Datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,   
    zoom_range=(0.95,0.95),
    horizontal_flip=True,
    vertical_flip=True,
    data_format='channels_last',
    validation_split=0.2,
    dtype = tf.float32
)
# read data from the dataset
trainGen = Datagen.flow_from_directory(
    'data/',
    target_size=(256,256),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True,
    subset='training',
    seed=123,
)


# create the cnn model


model = keras.Sequential(
    [
        layers.Reshape( ( 256, 256, 1 ) ),
        layers.Input( ( 256, 256, 1 ) ),
        layers.Conv2D( 128, ( 3, 3 ), padding='same', activation='relu'),
        layers.MaxPooling2D( pool_size = ( 2 , 2 ) ),
        layers.Conv2D( 64, ( 3, 3 ), padding='same', activation='relu'),
        layers.MaxPooling2D( pool_size = ( 2 , 2 ) ),
        layers.Conv2D( 32, ( 3, 3 ), padding='same', activation='relu'),
        layers.MaxPooling2D( pool_size = ( 2 , 2 ) ),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense( 16, activation='relu' ),
        layers.Dense( 2, activation='softmax' )
    ]
)

# model = keras.models.load_model('CNN_Model/TestModelLR1e-5_doubleLayers/')
# compile the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# fit the model
model.fit(trainGen, epochs=5, verbose=2)
# evaluate the model
# model.evaluate(trainGen, batch_size=32)
# save the model
model.save('CNN_Model/TestModelLR1e-5_doubleLayers_softmax_Dropout_Augmentation_newTestData')


