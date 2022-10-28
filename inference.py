import cv2
import tensorflow as tf
import numpy as np 
from tensorflow.keras.models import load_model
import librosa
from nomalise import MinMaxNormaliser
import soundfile as sf

scaler = MinMaxNormaliser(0,1)
model = load_model('models/autoencoder')
filename = input("Enter the name of the audio file: ")

x , sr = librosa.load(filename,sr=16000,duration=20)
S = np.abs(librosa.stft(x))
Xdb = librosa.amplitude_to_db(S)

a,b = Xdb.shape
h = np.zeros((1032-a,b))
v = np.zeros((1032,632-b))
result = np.vstack([Xdb,h])
result = np.hstack([result,v])
result = scaler.normalise(result)

res = model.predict(np.expand_dims(result,axis=0))
res = scaler.denormalise(res)

res = res[0].reshape(1032,632)
res = res[:1025,:626]

S2 = librosa.db_to_amplitude(res)
y_inv = librosa.griffinlim(S2)

sf.write(file='output.wav', data=y_inv, samplerate=sr)