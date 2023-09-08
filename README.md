# Histopathology Image Classification

This repository contains code and documentation for a histopathology image classification task. The code includes data preprocessing, model architecture, training, and evaluation. The following sections provide an overview of the repository's structure and functionality.

## Table of Contents

- [Imports](#imports)
- [Data Augmentation and Dataset Separation](#data-augmentation-and-dataset-separation)
- [The Given Model Architecture](#the-given-model-architecture)
- [Training Phase](#training-phase)
- [Plot of the Loss and Accuracy of the Given Model](#plot-of-the-loss-and-accuracy-of-the-given-model)
- [Confusion Matrix](#confusion-matrix)
- [Changing the Network Structure](#changing-the-network-structure)

## Imports

Import the required libraries and dependencies.

```python
import cv2
import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pickle
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
```

---

## Data Augmentation and Dataset Separation

This section of the code performs data augmentation and separates the dataset into training and test sets. Data augmentation helps increase the diversity of the dataset, which can improve model performance.

1. **Image Processing**: Images are processed to create augmented versions. These augmentations include horizontal flipping, histogram equalization, rotation, translation, and shearing.

2. **Counting**: The total number of color images is counted, and this count is printed.

3. **Dataset Separation**: The dataset is split into training and test sets using the `train_test_split` function. The training set contains 80% of the data, while the test set contains 20%.

```python
# Path to dataset
dataset_path = '/content/drive/MyDrive/image processing/dataset/dataset'

# A new directory to store augmented images
augmented_path = '/content/drive/MyDrive/image processing/augmented images'
os.makedirs(augmented_path, exist_ok=True)

# Lists to store image paths
image_paths = []

# Iterating through the images in the dataset
# (Code for data augmentation is here, refer to the original code)

# Counting the total number of color images
total_color_images = sum(1 for path in image_paths if cv2.imread(path).ndim > 2)
print("Total color images after data augmentation:", total_color_images)

# Split the dataset into training and test sets
train_paths, test_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

# Creating directories for the training and test sets
train_path = '/content/drive/MyDrive/image processing/train set'
test_path = '/content/drive/MyDrive/image processing/test set'
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Copy the training images to the train dataset folder
# Copy the test images to the test dataset folder
# (Code for copying images is here, refer to the original code)
```

---

## The Given Model Architecture

This section defines the architecture of the given model. The model is a sequential neural network with convolutional and fully connected layers.

```python
model = Sequential()

# Layer parameters
FS = (3, 3)  # Filter size
MP_size = (2, 2)  # Max-pooling size
NoF = 48  # Number of filters
P = 'same'  # Padding
activation = 'relu'  # Activation function

# Convolutional layers
layer_configs = [(2, 2, 2), (1, 1, 2), (5, 1, 1), (3, 1, 1), (3, 3, 1)]
for i, (s, strides, pooling) in enumerate(layer_configs, start=1):
    model.add(Conv2D(NoF, FS, strides=strides, padding=P, activation=activation, input_shape=(257, 257, 3)))
    if pooling:
        model.add(MaxPooling2D(pool_size=MP_size, strides=2))

# Flatten the tensor output
model.add(Flatten())

# Add a Fully Connected layer
model.add(Dense(1000, activation='relu'))
# Add a Dropout layer
model.add(Dropout(0.5))

# Add an output layer
model.add(Dense(2, activation='softmax'))

# Model summary
model.summary()
```

---

## Training Phase

This section covers the training phase of the model. The steps include data preparation, model compilation, early stopping, training, and evaluation.

```python
# The image size
image_size = (257, 257)

# Prepare data
# (Code for data preparation is here, refer to the original code)

# Compile the model
# (Code for model compilation is here, refer to the original code)

# Early stopping
# (Code for early stopping is here, refer to the original code)

# Training the model with early stopping
# (Code for model training is here, refer to the original code)

# Evaluating on the test data
# (Code for model evaluation is here, refer to the original code)

# Save the model's training history to a file using pickle
with open('/content/drive/MyDrive/image processing/history.pickle', 'wb') as file:
    pickle.dump(history.history, file)
```

---

## Plot of the Loss and Accuracy of the Given Model

This section loads the training history of the model from a file and plots the loss and accuracy over epochs.

```python
# Load the history object from the file
with open('/content/drive/MyDrive/image processing/history.pickle', 'rb') as file:
    history = pickle.load(file)

# Plot of model's loss and accuracy
# (Code for plotting loss and accuracy is here, refer to the original code)
```

---

## Confusion Matrix

This section computes the confusion matrix and performance metrics of the model on the test data and visualizes the confusion matrix.

```python
# Predictions on the test data
# (Code for making predictions is here, refer to the original code)

# Confusion matrix
# (Code for calculating confusion matrix is here, refer to the original code)

# Performance metrics
# (Code for calculating performance metrics is here, refer to the original code)

# Plot of confusion matrix
# (Code for plotting confusion matrix is here, refer to the original code)
```

---

## Changing the Network Structure

This section describes changes made to the network structure, including adding layers and altering batch size. It also includes the updated model's training and evaluation.

```python
# Modified model architecture with additional layers
# (Code for the modified model is here, refer to the original code

)

# Compile the modified model
# (Code for compiling the modified model is here, refer to the original code)

# Train the modified model
# (Code for training the modified model is here, refer to the original code)

# Evaluate the modified model
# (Code for evaluating the modified model is here, refer to the original code)

# Save the training history of the modified model
with open('/content/drive/MyDrive/image processing/history_modified.pickle', 'wb') as file:
    pickle.dump(history.history, file)
```

---

In conclusion, this repository provides code and documentation for histopathology image classification tasks, including data augmentation, model architecture, training, and evaluation. It also explores the impact of changing the network structure on model performance. You can use this as a reference for similar image classification projects.
