import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import warnings
import pygame
from sklearn.preprocessing import OneHotEncoder
warnings.filterwarnings('ignore')
#import the data to labels and paths
paths = []
labels = []
for dirname, _, filenames in os.walk('/Users/akselituominen/Desktop/TESS Toronto emotional speech set data'):
    for filename in filenames:
        if filename.startswith('.'):  # Skip hidden files like .DS_Store
            continue
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        if label.lower() == 'store':
            print(f"Found 'store' label in file: {filename}")
        labels.append(label.lower())
print('Dataset Loaded')

#Create a dataframe
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
df.head()

#Check the dataframe, 400 of each category so 2800 in total
print(df['label'].value_counts())

#Now we define functions for waveplot and spectrogram
#Waveplot to view the waveform of the audio
def waveplot(data, sr, emotion):
    plt.figure(figsize=(10,4))
    plt.title(emotion,size = 20)
    librosa.display.waveshow(data,sr=sr)
    plt.show()
#Spectrogram to view the frequency of the audio
def spectrogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11,4))
    plt.title(emotion,size = 20)
    librosa.display.specshow(xdb,sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()

#Now we define the emotions
#Im using pygame to play the audio
# Initialize Pygame mixer
pygame.mixer.init()


#Now we define the emotions
#This creates waveplots, spectrograms and plays the audio for every emotion
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
for emotion in emotions:
    path = df[df['label'] == emotion]['speech'].values[0]
    data, sampling_rare = librosa.load(path)
    waveplot(data, sampling_rare, emotion)
    spectrogram(data, sampling_rare, emotion)
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    # Keep the program running until the audio finishes playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(3)

#Feature extraction
#We will extract the features from the audio
#We will use librosa to extract the features, Mel Frequency Cepstral Coefficients (MFCC)
def excttactMCFF(filename):
    #Audio clip length 3 sec and start at 0.5 sec
    y, sr = librosa.load(filename,duration=3,offset=0.5)
    #Extracting MFCC features(40) and taking mean
    mfcc = np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
    return mfcc

#Save the features to a new dataframe
X_mfcc = df['speech'].apply(lambda x : excttactMCFF(x))
X = [x for x in X_mfcc]
X = np.array(X)
#X shape is not (2800, 40)

#Input split, add a new dimension for ML model
X = np.expand_dims(X,-1)

#We use OneHotEncoder to encode the labels
#It's used to convert categorical data into a format that can be provided to machine learning algorithms to improve prediction results.
#Turns categories to binary so it is easier for the model to understand

enc = OneHotEncoder()
y = enc.fit_transform(df[['label']]).toarray()
#Y Shape is (2800, 7), 7 nums of categories, 2800 nums of audio files

#Now lets create the LSTM model
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout

model = Sequential([
    #256 neurons, return only last output
    LSTM(256, return_sequences=False, input_shape=(40,1)),
    #Prevent overfitting with dropouts
    Dropout(0.2),
    #Fully connected 'Dense' layer with Rectified Linear Unit activation function
    Dense(128, activation = 'relu'),
    Dropout(0.2),
    Dense(64, activation = 'relu'),
    Dropout(0.2),
    #Output layer with 7 neurons for 7 categories and softmax activation function, softmax used to output probability distribution
    Dense(7, activation = 'softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

#Train the model
history = model.fit(X,y,epochs=50,batch_size=64,validation_split=0.2)


#Lets visualize the results
epochs = [i for i in range(50)]
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs,acc,'r',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#Save the model
model.save('model.h5')