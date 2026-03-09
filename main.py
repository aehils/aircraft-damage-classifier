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
from keras.src.legacy.preprocessing.image import ImageDataGenerator

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
    folder_name = "aircraft_damage_dataset_v1"   # name of the top-level folder inside .tar
    # actually download the .tar

    # if dataset folder already exists...
    if os.path.exists(folder_name):
        # skip download, no need to 
        print(f"Data is available locally at the existing folder: {folder_name}. Download cancelled.")
    else:
        urllib.request.urlretrieve(url, tar_filename)
        print(f"\nDownloaded {tar_filename} – extracting...")
        # Extract the contents of the tar file
        with tarfile.open(tar_filename, "r") as tar_ref:
            tar_ref.extractall()  # This will extract to the current directory
            print(f"Extracted {tar_filename} successfully.")

    # defining the directories for train, test and validation splits
    train_dir = os.path.join(folder_name, 'train')
    test_dir = os.path.join(folder_name, 'test')
    valid_dir = os.path.join(folder_name, 'valid')

    return train_dir, valid_dir, test_dir

def main():
    
    # get data
    train_dir, valid_dir, test_dir = fetch_dataset()

    # create ImageDataGenerators to preprocess each data split
    train_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        seed=seed_value,
        class_mode='binary',
        shuffle=True
    )
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        seed=seed_value,
        class_mode='binary',
        shuffle=False
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        seed=seed_value,
        shuffle=False,
        class_mode='binary'
    )

    # loading pre-train VGG16 model
    base_model = VGG16(include_top=False, weights='imagenet', 
                       input_shape=(img_rows, img_cols, 3))

if __name__ == '__main__':
    main()