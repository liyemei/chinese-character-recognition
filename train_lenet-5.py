# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 20:41:41 2018

@author: dxq
"""
from __future__ import print_function
import os
import keras
from keras.callbacks import History 
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Model, load_model
from keras.callbacks import History 
data_dir = './data'
train_data_dir = os.path.join(data_dir, 'train')
test_data_dir = os.path.join(data_dir, 'test')
from keras.models import Sequential
from keras.layers import Dense 

# dimensions of our images.
img_width, img_height = 28, 28
charset_size = 10
nb_validation_samples = 300
nb_samples_per_epoch = 512
nb_nb_epoch = 200;
 
def train(model):
  train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1
  )
  test_datagen = ImageDataGenerator(rescale=1./255)
 
  train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    color_mode="grayscale",
    class_mode='categorical')
  validation_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    color_mode="grayscale",
    class_mode='categorical')
  opt = keras.optimizers.rmsprop(lr=0.001, decay=0.0005)
  model.compile(loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'])
  history=model.fit_generator(train_generator,
    samples_per_epoch=nb_samples_per_epoch,
    nb_epoch=nb_nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)
  
  return history

model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5,5), padding='valid', input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(charset_size, activation='softmax'))


history=train(model)
train_loss=history.history['acc']
val_loss=history.history['val_acc']
import matplotlib.pyplot as plt

fig1=plt.gcf() 
plt.plot(train_loss, 'r-',label='train_acc')
plt.plot(val_loss, 'b--',label='val_acc')
plt.legend(loc = 'best')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
fig1.savefig('lenet_result.png')

model.save_weights('lenet__model.h5')