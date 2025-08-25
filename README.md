# AMERICAN SIGN LANGUAGE

## Project Overview

This project focuses on recognizing American Sign Language (ASL) hand signs using Convolutional Neural Networks (CNNs). The model is trained on labeled images of hand gestures and classifies them into corresponding ASL alphabets.

## Key Features

- Built using Deep Learning (CNNs) for high-accuracy image classification.
- Falls under Computer Vision and Gesture Recognition domains.
- Enables real-time ASL recognition for assistive communication.
- Contributes to AI for Accessibility by bridging the gap between spoken and sign language.

## Domains Covered

- Deep Learning
- Computer Vision
- Image Classification
- Gesture Recognition

## How It Works

### 1. Dataset Preparation

- Collected a dataset of ASL hand signs (A–Z alphabets and/or numbers).
- Images were resized, normalized, and augmented (rotation, flipping, zoom) to improve generalization.

### 2. Model Architecture

- Implemented a Convolutional Neural Network (CNN)
- Convolution + MaxPooling layers for feature extraction.
- Dropout layers to prevent overfitting.
- Batch Normalization layers for faster convergence and better generalization.

### 3. Training

- Model trained on the dataset using categorical cross-entropy loss and Adam optimizer.
- Achieved high accuracy in recognizing different ASL gestures.

### 4. Evaluation

- Tested the model on unseen images to validate performance.
- Evaluated using accuracy, precision, recall, and confusion matrix.

### 5. Prediction

- Input: Hand sign image.
- Output: Predicted ASL alphabet/gesture.

### 6. Result

<img width="1224" height="464" alt="image" src="https://github.com/user-attachments/assets/44d7c183-2bfc-4ec4-ae21-ea9eb56013af" />


<img width="265" height="56" alt="image" src="https://github.com/user-attachments/assets/e4a5e71a-ccaa-4ad7-9ca7-accc279cb391" />

### 7. Limitations

- It predicts only static alphabet signs.
- It does not include predictions of J and Z which are motion based signs as per American Sign Language usage.
- Background disturbances and noise tend to reduce the prediction accuracy at times.

