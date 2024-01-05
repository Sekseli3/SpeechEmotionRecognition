# Emotion Recognition from Audio Files

This project uses a Long Short-Term Memory (LSTM) model to classify emotions from audio files. The model is trained on a dataset of audio files, each labeled with one of seven emotions: 'angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', and 'sad'.
Data from kaggle: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess?resource=download 
Each emotion gets plotted for visual reasons, does not really do anything in training:
<img width="966" alt="Screenshot 2024-01-05 at 19 42 03" src="https://github.com/Sekseli3/SpeechEmotionRecognition/assets/120391401/78dc92ab-0741-4293-b9c2-9bc98a4431eb">
<img width="965" alt="Screenshot 2024-01-05 at 19 42 09" src="https://github.com/Sekseli3/SpeechEmotionRecognition/assets/120391401/e6ea18d1-eab8-4473-88d5-9007c904fb1a">


## Model Architecturem

The model consists of an LSTM layer followed by three Dense layers. Dropout is applied after the LSTM layer and each Dense layer to prevent overfitting. The output layer uses the softmax activation function to output a probability distribution over the seven emotion classes.

The model is compiled with the Adam optimizer and the categorical cross-entropy loss function.

## Training

The model is trained for 50 epochs with a batch size of 64. 20% of the data is used for validation during training.

## Results

The training and validation accuracy are plotted after each epoch. Right not the results are not yet as good as i would like them to be.

## Usage

To use the trained model to classify an emotion from an audio file, load the model from 'model.h5' and use the `predict` method.

## Dependencies

- Keras
- Librosa
- Matplotlib
- Numpy

## TODO
Get better results, the model is overfitting i think
