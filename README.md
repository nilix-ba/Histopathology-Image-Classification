# Histopathology Image Classification

This repository contains code for an image classification task on histopathology images. The code includes data augmentation, dataset separation, model architecture, training, and evaluation. The original model is compared to a modified model to assess their performance.

## Table of Contents

- [Introduction](#introduction)
- [Data Augmentation and Dataset Separation](#data-augmentation-and-dataset-separation)
- [The Given Model Architecture](#the-given-model-architecture)
- [Training Phase](#training-phase)
- [Plot of Loss and Accuracy](#plot-of-loss-and-accuracy)
- [Confusion Matrix](#confusion-matrix)
- [Modifying the Network Structure](#modifying-the-network-structure)
- [Model Comparison](#model-comparison)

## Introduction

This code is developed for histopathology image classification. It performs the following tasks:

1. Data augmentation to increase the dataset size and diversity.
2. Separation of the dataset into training and test sets.
3. Creation of a convolutional neural network (CNN) model for image classification.
4. Training and evaluation of the model.
5. Comparison of the performance between the original and modified models.


## Data Augmentation and Dataset Separation

The code performs data augmentation and separates the dataset into training and test sets.

- Data augmentation includes flipping, histogram equalization, rotation, translation, and shearing of images.
- The dataset is separated into training and test sets with an 80-20 split.

## The Given Model Architecture

The given model architecture consists of:

1. Convolutional layers with ReLU activation.
2. Max-pooling layers.
3. A fully connected layer with 1000 units and ReLU activation.
4. A dropout layer with a dropout rate of 0.5.
5. An output layer with 2 units and softmax activation.

The model is compiled using binary cross-entropy loss, the Adam optimizer with a learning rate of 0.0001, and accuracy as the metric. Early stopping is applied to prevent overfitting.

## Training Phase

In the training phase:

- The image size is set to (257, 257).
- Data is loaded and preprocessed.
- The model is compiled and trained with early stopping.
- The model is evaluated on the test data, and loss and accuracy are reported.
- The training history is saved to a file using pickle.

## Plot of Loss and Accuracy

Plots of model loss and accuracy during training are generated from the saved training history.

## Confusion Matrix

A confusion matrix is generated to evaluate model performance. Metrics such as accuracy, precision, recall, and specificity are calculated and displayed. The confusion matrix is also visualized as a heatmap.

## Modifying the Network Structure

The code modifies the network structure by adding an extra fully connected layer and a dropout layer after the first fully connected layer. Additionally, the batch size is changed to 200 for training.

## Model Comparison

The performance of the original and modified models is compared based on accuracy, precision, recall, specificity, and the confusion matrix. The results indicate that the original model outperforms the modified model in this task.

For further details, refer to the code and comments within the Jupyter notebook or script.
