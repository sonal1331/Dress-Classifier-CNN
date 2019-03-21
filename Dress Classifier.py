# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:37:27 2019

@author: Sonal
"""

import tensorflow as tf
from keras.models import Sequential
classifier=Sequential()

from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os
os.chdir(r"C:\Users\Dell\Desktop")

classifier= Sequential()
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow_from_directory('train',
                                               target_size=(64,64),
                                               batch_size=10,
                                               class_mode='binary')
test_set=train_datagen.flow_from_directory('test',
                                               target_size=(64,64),
                                               batch_size=10,
                                               class_mode='binary')
classifier.fit_generator(training_set,
                         samples_per_epoch=1430,
                         nb_epoch=20,
                         validation_data=test_set,
                         nb_val_samples=572)
