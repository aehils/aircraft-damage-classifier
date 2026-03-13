#!/usr/bin/env python3 

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

import tarfile, urllib.request, os

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

# Function to plot a single image and its prediction
def plot_image_with_title(image, model, true_label, predicted_label, class_names):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)

    # Convert labels from one-hot to class indices if needed, but for binary labels it's just 0 or 1
    true_label_name = class_names[true_label]  # Labels are already in class indices
    pred_label_name = class_names[predicted_label]  # Predictions are 0 or 1

    plt.title(f"True: {true_label_name}\nPred: {pred_label_name}")
    plt.axis('off')
    plt.show()

# Function to test the model with images from the test set
def test_model_on_image(test_generator, model, index_to_plot=0):
    # Get a batch of images and labels from the test generator
    test_images, test_labels = next(test_generator)

    # Make predictions on the batch
    predictions = model.predict(test_images)

    # In binary classification, predictions are probabilities (float). Convert to binary (0 or 1)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    # Get the class indices from the test generator and invert them to get class names
    class_indices = test_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}  # Invert the dictionary

    # Specify the image to display based on the index
    image_to_plot = test_images[index_to_plot]
    true_label = test_labels[index_to_plot]
    predicted_label = predicted_classes[index_to_plot]

    # Plot the selected image with its true and predicted labels
    plot_image_with_title(image=image_to_plot, model=model, true_label=true_label, predicted_label=predicted_label, class_names=class_names)

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

    # loading pre-trained VGG16 model
    base_model = VGG16(include_top=False, weights='imagenet', 
                       input_shape=(img_rows, img_cols, 3))
    
    output = base_model.layers[-1].output
    output = keras.layers.Flatten()(output)
    base_model = Model(base_model.input, output)
    # freeze base layers of VGG16
    for layer in base_model.layers:
        layer.trainable = False
    
    # build custom model architecture
    model = Sequential()
    model.add(base_model)
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
        # compile!
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=valid_generator
    )
    train_history = model.history.history   # access the training history

    # plot the LOSS for both training and validation
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title("Training Loss")  # training
    plt.ylabel("Loss")
    plt.xlabel('Epoch')
    plt.plot(train_history['loss'])
    # ––––––––––––––
    plt.subplot(1, 2, 1)
    plt.title("Validation Loss")    # validation
    plt.ylabel("Loss")
    plt.xlabel('Epoch')
    plt.plot(train_history['val_loss'])
    plt.show()

    # plot the ACCURACY for both training and validation sets
    plt.figure(figsize=(5, 5))
    plt.plot(train_history['accuracy'], label='Training Accuracy')
    plt.plot(train_history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # evaluate model performance on test data
    test_loss, test_accuracy = model.evaluate(test_generator,
                                              steps=(test_generator.samples // test_generator.batch_size))
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # test model on selected image
    test_model_on_image(test_generator=test_generator,
                        model=model,
                        index_to_plot=1)
    

if __name__ == '__main__':
    main()