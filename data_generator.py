import numpy as np 
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from normalise import MinMaxNormaliser

class CustomDataGenerator(Sequence):
    
    def __init__(self, df, X_col, y_col, batch_size, shuffle=True):
    
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(self.df)
        self.scaler = MinMaxNormaliser(0,1)
        
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            
    def __len__(self):
        return self.n // self.batch_size
    
    def __getitem__(self,index):
    
        batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size,:]
        X1, X2 = self.__get_data(batch)        
        return X1, X2
    
    def __padding(self, spec):
        a,b = spec.shape
        h = np.zeros((1032-a,b))
        v = np.zeros((1032,632-b))
        result = np.vstack([spec,h])
        result = np.hstack([result,v])
        return result
    
    def __extract_spectrograms(self,audio):
        x , sr = librosa.load(audio,sr=16000)
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(np.abs(X),)
        return Xdb
    
    def __get_data(self,batch):
        
        X1, X2 = list(), list()
        src_audios = batch[self.X_col].tolist()
        target_audios = batch[self.y_col].tolist()
        
        for src,target in zip(src_audios,target_audios):
            input_spec = self.__extract_spectrograms(src)
            output_spec = self.__extract_spectrograms(target)
            input_spec = self.__padding(input_spec).reshape(1032,632,1)
            output_spec = self.__padding(output_spec).reshape(1032,632,1)
            X1.append(input_spec)
            X2.append(output_spec)
            
        X1, X2 = np.array(X1), np.array(X2)
        X1 = self.scaler.normalise(X1)
        X2 = self.scaler.normalise(X2)
        
        return X1, X2