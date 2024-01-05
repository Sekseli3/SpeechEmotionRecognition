# Emotion Recognition from Audio Files

This project uses a Long Short-Term Memory (LSTM) model to classify emotions from audio files. The model is trained on a dataset of audio files, each labeled with one of seven emotions: 'angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', and 'sad'.

## Model Architecture

The model consists of an LSTM layer followed by three Dense layers. Dropout is applied after the LSTM layer and each Dense layer to prevent overfitting. The output layer uses the softmax activation function to output a probability distribution over the seven emotion classes.

The model is compiled with the Adam optimizer and the categorical cross-entropy loss function.

## Training

The model is trained for 50 epochs with a batch size of 64. 20% of the data is used for validation during training.

## Results

The training and validation accuracy are plotted after each epoch.

## Usage

To use the trained model to classify an emotion from an audio file, load the model from 'model.h5' and use the `predict` method.

## Dependencies

- Keras
- Librosa
- Matplotlib
- Numpy

## TODO
Get better results, the model is overfitting i think
