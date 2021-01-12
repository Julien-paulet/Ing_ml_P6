#!/usr/bin/env python
# coding: utf-8

import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
from keras.preprocessing import image
import logging
import cv2

# Silence Tensorlfow log on logging, annoying while in prod
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# If this is not set, bug because tf takes too much memory
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Picture path init:
pathPic = str(sys.argv[1])

imagePath = pathPic

width, height = 224,224
imageSize = (width, height)

# Preparing the image
im = image.load_img(imagePath)
im = im.resize(imageSize)
im = image.img_to_array(im)
im = (np.expand_dims(im, 0))

# Load model
print("Loading Model, please wait...")
model = tf.keras.models.load_model('modelBase/')

print("Prediction on its way !")
pred = model.predict(im)

predicted_class_indices=np.argmax(pred,axis=1)

# Open csv files containing index of each category
classes = pd.read_csv('classes.csv', header=None).set_index(0)

prediction = classes.loc[predicted_class_indices,:].values.item()
prediction = prediction.split('-')[1]

print("Prediction's done ->")
print("Class of the picture is: ", prediction)

# Killing cuda process
import keras
from numba import cuda
# Clearing session
keras.backend.clear_session()
# Releasing gpu memory directly with cuda
cuda.select_device(0)
cuda.close()

