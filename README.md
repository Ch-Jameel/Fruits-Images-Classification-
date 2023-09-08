# Fruit Classification with TensorFlow

This project is an example of fruit classification using TensorFlow, specifically focusing on creating a Convolutional Neural Network (CNN) model to classify different types of fruits.

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Inference](#inference)
- [Saving the Model](#saving-the-model)

## Introduction

In this project, we demonstrate how to build a fruit classification model using TensorFlow. We use a CNN architecture to train a model that can classify various types of fruits based on input images. The dataset used for training contains six classes: acai, cupuacu, graviola, guarana, pupunha, and tucuma.

## Getting Started

Before running the code, ensure that you have all the necessary libraries and dependencies installed. Import TensorFlow and other required libraries at the beginning of your script. This project assumes you are working in a Jupyter Notebook or similar environment.

```python
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML
```

## Data Preparation

To load and preprocess the data, we use TensorFlow's `image_dataset_from_directory` API. This API loads the images from the specified directory and organizes them into a TensorFlow dataset. We specify constants such as batch size, image size, channels, and epochs for data processing and training.

```python
BATCH_SIZE = 5
IMAGE_SIZE = 256
CHANNELS = 3
EPOCHS = 50

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "/content/drive/MyDrive/Datasets/ds_frutas_am/train",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)
```

## Model Architecture

Our CNN model consists of convolutional layers, max-pooling layers, and fully connected layers. Before feeding the images into the network, we add layers for resizing and normalization.

```python
# Model Architecture
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 6

model = models.Sequential([
    # Add resizing and rescaling layers here
    # Convolutional and max-pooling layers
    # Fully connected layers
])
```

## Training

We compile the model with appropriate loss and optimization functions and train it on the training dataset.

```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=10,
    verbose=1,
    validation_data=val_ds
)
```

## Inference

To make predictions, we define a function to preprocess and predict on sample images from the test dataset.

```python
def predict(model, img):
    # Preprocess the image
    # Make predictions
    return predicted_class, confidence
```

## Saving the Model

You can save the trained model for later use.

```python
model.save("fruit_classification.h5")
```

Feel free to explore this project further and modify the code as needed. Happy coding!

---

Make sure to replace the code placeholders with actual links to your dataset, and update the model architecture section with the appropriate layers and configurations used in your project.
