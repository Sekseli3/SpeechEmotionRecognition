from keras.models import load_model
import numpy as np
import librosa

model = load_model('model.h5')

# Load the audio file
filename = "/Users/akselituominen/Desktop/sound4.wav"
y, sr = librosa.load(filename, duration=3, offset=0.5)

# Extract the MFCCs and take the mean
mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

# Expand the dimensions to match the input shape expected by the model
mfcc = np.expand_dims(mfcc, axis=-1)


# Make a prediction
prediction = model.predict(np.array([mfcc]))
categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
# The prediction is a 7-element vector of probabilities. To get the predicted class, take the index of the highest probability.
predicted_class = np.argmax(prediction)

print("The predicted class is:", categories[predicted_class])
print("The predictions for each category are:")
for i in range(len(categories)):
    category = categories[i]
    probability = prediction[0][i]
    print(category, ":", probability)