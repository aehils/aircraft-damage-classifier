#!/usr/bin/env python3 

import zipfile
import keras
import random

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.preprocessing import image

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# set seeds for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# setting batch size, epochs and input image shape
batch_size = 32
epochs = 5
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)
