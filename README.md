# Handwritten-Digit-Recognition-Neural-Network
A Python project that employs TensorFlow to recognize handwritten digits (0-9)

Author: Rambod Azimi

This repository contains a Python script that utilizes a neural network to recognize handwritten digits (0-9) using the TensorFlow framework. The script demonstrates the complete process, from loading the dataset to training the model and making predictions.

## Table of Contents

- [Requirements](#requirements)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Compilation and Training](#compilation-and-training)
- [Making Predictions](#making-predictions)
- [Results](#results)

## Requirements

Before using this code, ensure you have the following dependencies installed:

- Python (3.x recommended)
- TensorFlow
- NumPy
- OS

You can install TensorFlow and NumPy using the following commands:

```bash
pip install tensorflow
pip install numpy

```

## Dataset

![model architecture](https://i.ibb.co/9tDyJ2n/Screenshot-2023-08-10-at-6-59-08-PM.png)


The dataset used in this script contains 5000 training examples of handwritten digits (0-9). Each training example is a 20x20 pixel grayscale image of the digit, with each pixel represented by a floating-point number indicating grayscale intensity.

## Usage

1. Clone this repository

git clone https://github.com/rambodazimi/Handwritten-Digit-Recognition-Neural-Network.git

2. Run the Python script

python Multiclass Neural Network.py

## Model Architecture

![model architecture](https://i.ibb.co/0K0KBKM/C2-W2-Assigment-NN.png)


The neural network model is defined as follows:

- Input Layer (400 units)

- Hidden Layer 1 (25 units, ReLU activation)

- Hidden Layer 2 (15 units, ReLU activation)

- Output Layer (10 units, linear activation)

## Compilation and Training

The model is compiled using the Adam optimizer and the sparse categorical cross-entropy loss function. It is trained on the training dataset for 70 epochs.

## Making Predictions

An example image of the digit "2" is used for prediction. The model predicts the digit using the softmax function to convert the output into a probability distribution.

## Results

The predicted digit is printed in the console output. Feel free to modify the code to test other images or enhance the model's performance.
