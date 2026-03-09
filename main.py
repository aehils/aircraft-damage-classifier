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

import tarfile, urllib.request, os, shutil

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

def fetch_dataset():
    # URL of the tarfile w the data
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar"

    # define location to save the data
    tar_filename = "aircraft_damage_dataset_v1.tar"
    extracted_folder = "aircraft_damage_dataset_v1"   # where extracted contents will go
    # actually download the .tar
    urllib.request.urlretrieve(url, tar_filename)
    print(f"\nDownloaded {tar_filename} – extracting...")

    # if dataset folder already exists...
    if os.path.exists(extracted_folder):
        print(f"The folder '{extracted_folder}' already exists. Removing the existing folder.")
        
        # remove it, keep things simple
        shutil.rmtree(extracted_folder)
        print(f"Removed the existing folder: {extracted_folder}")

    # Extract the contents of the tar file
    with tarfile.open(tar_filename, "r") as tar_ref:
        tar_ref.extractall()  # This will extract to the current directory
        print(f"Extracted {tar_filename} successfully.")

    # defining the directories for train, test and validation splits
    extract_path = extracted_folder
    train_dir = os.path.join(extract_path, 'train')
    test_dir = os.path.join(extract_path, 'test')
    valid_dir = os.path.join(extract_path, 'valid')

    return train_dir, test_dir, valid_dir