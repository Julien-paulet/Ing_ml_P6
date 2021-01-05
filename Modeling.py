#!/usr/bin/env python
# coding: utf-8
import os
import sys
import pandas as pd
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import cv2
import warnings
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from config_file_generator import create_config_file
tf.random.set_seed(42)
warnings.filterwarnings("ignore")

# If this is not set, bug because tf takes too much memory
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Locked variable, does not change with config file
path = '/home/mlmaster/Code/Ing_ml_P6/dataset/'
train = 'Train/'
val = 'Validation/'
test = 'Test/'
fullPathTrain = path + train
fullPathVal = path + val
fullPathTest = path + test

# For Logs
save = "/home/mlmaster/Code/Ing_ml_P6/logs/"

#setting the path to the directory containing the pics
categories = []
for i in os.listdir(fullPathTrain):
    categories.append(i)

def load_data(which='Train', categories=categories):
    
    """Function that loads data and resize the images"""
    
    if which == 'Train':
        path = fullPathTrain
        print('Loading Train Data')
    elif which == 'Val':
        path = fullPathVal
        print('Loading Validation Data')
    elif which == 'Test':
        path = fullPathTest
        print('Loading Test Data')
    else:
        print('error, which must be either : Train, Val, Test')
    data = []
    label = []
    for i in categories:
        for img in os.listdir(path + i):
            pic = cv2.imread(os.path.join(path + i,img))
            pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
            pic = cv2.resize(pic,(img_height, img_width))
            data.append([pic])
            label.append(i)
    print(len(data), " Images loaded")
    # Formating data
    data = np.array(data)
    data = data.reshape((data.shape[0], img_height, img_width, 3))
    data = data.astype('float32')
    label = np.array(label)
    label = [i.split('-')[1] for i in label]
    
    return data, label

# Init variables
min_index = int(sys.argv[1])
max_index = int(sys.argv[2])
config = create_config_file(min_index, max_index)
config = config.set_index('Index')

# -------------------------------------------
for i in config.index:
    # Global
    index = i

    # For data
    img_height = config.loc[i, 'Img Height']
    img_width = config.loc[i, 'Img Width']
    zca_whitening = config.loc[i, 'Zca Whitening']
    horizontal_flip = config.loc[i, 'Horizontal Flip']

    # For model
    model_type = config.loc[i, 'Model Type']
    batch_size = config.loc[i, 'Batch Size']
    learning_rate = config.loc[i, 'Learning Rate']
    epochs = config.loc[i, 'Epochs']
    steps_per_epoch = int(12000 / (batch_size))
    validation_steps = int(1200 / (batch_size))

    # Building Dataset for logs
    dataset = [index, img_height, img_width, batch_size,
               learning_rate, epochs, steps_per_epoch,
               validation_steps, zca_whitening, horizontal_flip,
               model_type]

    dataset = pd.DataFrame(dataset).T
    dataset = dataset.rename(columns={0:'Index', 1:'Img Height', 2:'Img Width', 3:'Batch Size',
                                      4:'Learning Rate', 5:'Epochs', 6:'Steps Per Epoch',
                                      7:'validation Steps', 8:'Zca Whitening', 9:'Horizontal Flip',
                                      10:'Model Type'})

    # For model
    train_datagen =  ImageDataGenerator(
        zca_whitening=zca_whitening,
        horizontal_flip=horizontal_flip,
        rescale=1./255
    )

    val_datagen =  ImageDataGenerator(
        rescale=1./255
    )

    test_datagen =  ImageDataGenerator(
        rescale=1./255
    )

    # Fit if zca_whitening is True
    if zca_whitening == True:
        dataToFit, label = load_data(which='Train', categories=categories)
        dataToFit = dataToFit.reshape((dataToFit.shape[0], img_height, img_width, 3))
        dataToFit = dataToFit.astype('float32')
        train_datagen.fit(dataToFit)
        # This fail with error :
        # "ValueError: Too large work array required -- 
        # computation cannot be performed with standard 32-bit LAPACK."

    # Defining the generator
    print('Total number of images for "training":')
    train_generator = train_datagen.flow_from_directory(
                                    fullPathTrain,
                                    target_size = (img_height, img_width),
                                    batch_size = batch_size, 
                                    class_mode = "categorical")

    print('Total number of images for "validation":')
    val_generator = test_datagen.flow_from_directory(
                                 fullPathVal,
                                 target_size = (img_height, img_width),
                                 batch_size = batch_size,
                                 class_mode = "categorical",
                                 shuffle=False)

    print('Total number of images for "testing":')
    test_generator = test_datagen.flow_from_directory(
                                  fullPathTest,
                                  target_size = (img_height, img_width),
                                  batch_size = batch_size,
                                  class_mode = "categorical",
                                  shuffle=False)

    # Model Builder
    if model_type == 1:
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2),
                         activation='relu',
                         input_shape=(img_height,img_width,3)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        model.add(Dropout(0.5))
        model.add(Conv2D(32, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(16, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(8, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(len(categories), activation='softmax')) 

    elif model_type == 2:
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=(img_height,img_width,3)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(len(categories), activation='softmax'))

    elif model_type == 3:
        model = Sequential()
        model.add(Conv2D(8, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=(img_height,img_width,3)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(16, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(len(categories), activation='softmax'))

    elif model_type == 4:
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2),
                         activation='relu',
                         input_shape=(img_height,img_width,3)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        model.add(Dropout(0.5))
        model.add(Conv2D(32, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(len(categories), activation='softmax'))  

    # Callback
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                                mode='min',
                                                restore_best_weights=True)
        
    # Optmizer and compilation
    opt = Adam(lr=learning_rate, clipnorm=1.)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    # Model fit via generator
    history = model.fit(train_generator,
                        epochs=epochs,
                        validation_data=val_generator,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        callbacks=[callback],
                        verbose=1)

    # Ploting graph

    path_img = save + str(index) + '/'
    try:
        os.mkdir(path_img)
    except:
        print('This directory already exists, please make sure that you changed the index')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1,len(acc)+1)

    plt.figure()
    plt.plot(epochs, acc, 'b', label = 'Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(path_img + 'Accuracy.jpg')
    plt.plot()

    plt.figure()
    plt.plot(epochs, loss, 'b', label = 'Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(path_img + 'Loss.jpg')
    plt.plot()


    # Saving best scores
    val_acc = max(history.history['val_accuracy'])
    val_loss = max(history.history['val_loss'])

    dataset['Validation Accuracy'] = val_acc
    dataset['Validation Loss'] = val_loss

    try:
        logs = pd.read_csv(save + 'logs.csv')
        logs = pd.concat([logs, dataset])
        logs.to_csv(save + 'logs.csv')
    except:
        dataset.to_csv(save + 'logs.csv')

import keras
from numba import cuda
# Clearing session as we will train model many time
keras.backend.clear_session()
# Releasing gpu memory directly with cuda
cuda.select_device(0)
cuda.close()



