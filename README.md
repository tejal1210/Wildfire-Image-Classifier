# Wildfire Classifier Model

This repository contains a Convolutional Neural Network (CNN) model for classifying images related to wildfires. The model is trained to distinguish between images containing wildfires and images without wildfires.

## Dataset

The dataset used for training, validation, and testing the model is available in the `wildfire-prediction-dataset` directory. It includes three subdirectories:

- `train`: Contains images for training the model.
- `valid`: Contains images for validating the model during training.
- `test`: Contains images for evaluating the model after training.

## Model Architecture

The CNN model architecture is as follows:

- Input layer: Accepts images of size 350x350 pixels with 3 color channels.
- Convolutional layers:  One convolutional and  one max-pooling layer.
- Flatten layer: Flattens the output from convolutional layers into a 1D array.
- Dense layers: One dense layer with dropout for regularization.
- Output layer: Dense layer with softmax activation for classifying images into two classes (wildfire or non-wildfire).

## Training

The model is trained using the training dataset (`train`) and validated using the validation dataset (`valid`). During training, data augmentation techniques such as rotation, width and height shifting, rescaling, shearing, zooming, and horizontal flipping are applied to enhance model generalization.

Training parameters:
- Batch size: 256
- Number of epochs: 5
- Optimizer: Adam with a learning rate of 0.001
- Loss function: Categorical crossentropy

