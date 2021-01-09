#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import random

img_size = np.arange(120,260,20) # We only use images with same width and height here
zca_whitening = [False, False] # Currently does not work if zca is set to true
horizontal_flip = [True, False]
model_type = [3]
batch_size = np.arange(20, 60, 10)
learning_rate = [0.001, 0.0001]
epochs = np.arange(10, 50, 10)
rotation_range = np.arange(0,40)
width_shift_range = np.arange(0, 0.3, 0.1)
height_shift_range = np.arange(0, 0.3, 0.1)

def create_config_file(min_index, max_index):
    
    """Function to create a config array with the number of iteration that input the user.
    It creates that array by choosing randomly from the param init above"""
    
    index = np.arange(min_index, max_index+1)
    nbrIter = len(index)
    config_ = []
    for i in range(0, nbrIter):
        index_ = index[i]
        img_size_ = random.choice(img_size)
        zca_whitening_ = random.choice(zca_whitening)
        horizontal_flip_ = random.choice(horizontal_flip)
        model_type_ = random.choice(model_type)
        batch_size_ = random.choice(batch_size)
        learning_rate_ = random.choice(learning_rate)
        epochs_ = random.choice(epochs)
        rotation_range_ = random.choice(rotation_range)
        width_shift_range_ = random.choice(width_shift_range)
        height_shift_range_ = random.choice(height_shift_range)
        
        # Define img height and width with img size
        img_height_ = img_size_
        img_width_ = img_size_
        
        data = [index_, img_height_, img_width_,
                zca_whitening_, horizontal_flip_,
                model_type_, batch_size_, learning_rate_,
                epochs_, rotation_range_, width_shift_range_,
                height_shift_range_]
        
        config_.append(data)
        
    config_ = pd.DataFrame(config_, columns=['Index', 'Img Height', 'Img Width',
                                             'Zca Whitening', 'Horizontal Flip', 'Model Type',
                                             'Batch Size', 'Learning Rate', 'Epochs',
                                             'Rotation Range', 'Width Shift Range', 'Height Shift Range'])
    
    return config_

