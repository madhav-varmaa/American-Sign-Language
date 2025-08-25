# AMERICAN SIGN LANGUAGE

## Project Overview

This project focuses on recognizing American Sign Language (ASL) hand signs using Convolutional Neural Networks (CNNs). The model is trained on labeled images of hand gestures and classifies them into corresponding ASL alphabets.

## Key Features

- Built using Deep Learning (CNNs) for high-accuracy image classification.
- Falls under Computer Vision and Gesture Recognition domains.
- Enables real-time ASL recognition for assistive communication.
- Contributes to AI for Accessibility by bridging the gap between spoken and sign language.

## Domains Covered

-Deep Learning.
-Computer Vision.
-Image Classification.
-Gesture Recognition.

## How It Works

### 1. Dataset Preparation

Collected a dataset of ASL hand signs (A–Z alphabets and/or numbers).

Images were resized, normalized, and augmented (rotation, flipping, zoom) to improve generalization.

### 2. Model Architecture

Implemented a Convolutional Neural Network (CNN) with:

Convolution + MaxPooling layers for feature extraction.

Dropout layers to prevent overfitting.

Fully connected layers with Softmax for classification.

### 3. Training

Model trained on the dataset using categorical cross-entropy loss and Adam optimizer.

Achieved high accuracy in recognizing different ASL gestures.

### 4. Evaluation

Tested the model on unseen images to validate performance.

Evaluated using accuracy, precision, recall, and confusion matrix.

### 5. Prediction

Input: Hand sign image.

Output: Predicted ASL alphabet/gesture.
