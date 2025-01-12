# 🌸 Flower Type Recognition using Convolutional Neural Networks (CNN) 🌻

## Overview

The goal of this project is to develop a machine learning model capable of recognizing different flower types from images using Convolutional Neural Networks (CNN). CNNs are powerful deep learning models that excel in image classification tasks. In this project, the dataset contains images of five flower types: *dandelion*, *daisy*, *sunflower*, *tulip*, and *rose*. The CNN model is trained to classify these flowers based on the patterns and features extracted from the images.

## Dataset 📊

The dataset used in this project consists of images from five flower categories:

- 🌼 **Dandelion**
- 🌸 **Daisy**
- 🌻 **Sunflower**
- 🌷 **Tulip**
- 🌹 **Rose**

You can download the dataset [here](https://www.kaggle.com/datasets/senasudemir/flowers). 

Each image is preprocessed to a consistent size and normalized to ensure the model performs efficiently.

## Model Architecture 🧠

The model is a **Convolutional Neural Network (CNN)**, a powerful architecture commonly used for image recognition tasks. The network consists of multiple convolutional layers, activation functions, max-pooling, and dropout layers to capture relevant features and avoid overfitting. The output layer is a softmax classifier that predicts one of the five flower categories.

### Key Features:
- **Convolutional layers** for feature extraction
- **Activation functions (ReLU)** for non-linearity
- **Max-pooling layers** for dimensionality reduction
- **Dropout layers** to prevent overfitting
- **Softmax output layer** to classify images into one of the five categories

## Objective 🎯

The objective of this project is to build a CNN model that can accurately classify images of flowers into one of the five categories. The model aims to achieve high classification accuracy and demonstrate the effectiveness of deep learning in solving image classification tasks.

## Results 📈

The model achieved an accuracy of **80.13%** on the test dataset, indicating strong performance in classifying the five flower types. 


### Analysis of the Confusion Matrix:
- **Daisy:** The model correctly classified 198 daisy images, but there were misclassifications with *dandelion* (22) and *sunflower* (9).
- **Dandelion:** The dandelion class performed well with 179 correct predictions but had misclassifications with *daisy* (3) and *rose* (12).
- **Rose:** The model classified 157 rose images correctly, but some misclassifications occurred with *tulip* (18) and *sunflower* (13).
- **Sunflower:** The sunflower class showed strong performance with minimal misclassifications.
- **Tulip:** The tulip class showed a fair performance, with 133 correct predictions but misclassifications with *sunflower* and *rose*.

In summary, the model performs well with an accuracy of over **80%**, but there are areas for improvement in reducing misclassifications between similar flower types, especially for *tulip* and *sunflower*.

## Demo 🌐

You can test the model in real-time on Hugging Face Spaces!

[Flower Recognition Demo on Hugging Face](https://huggingface.co/spaces/Senasu/Flower_Recognition)
