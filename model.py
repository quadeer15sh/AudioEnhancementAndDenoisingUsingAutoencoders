import os
import numpy as np 
import pandas as pd 
import librosa
import matplotlib.pyplot as plt 
import seaborn as sns
import re
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dropout, Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D,Flatten, Dense, Input, Layer, Add, Reshape, Lambda
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from normalise import MinMaxNormaliser
from data_generator import CustomDataGenerator

noisy = '../input/noisyandcleanaudios/speaker_audios_new/noisy_audios'
clean = '../input/noisyandcleanaudios/speaker_audios_new/clean_audios'
data = pd.read_csv("../input/noisyandcleanaudios/speaker_audios_new/audios_meta_data.csv")

data.loc[data['noisy_audio']!=data['clean_audio'],'noisy_audio'] = data.loc[data['noisy_audio']!=data['clean_audio']]['noisy_audio'].apply(lambda x: os.path.join(noisy,x))
data.loc[data['noisy_audio']==data['clean_audio'],'noisy_audio'] = data.loc[data['noisy_audio']==data['clean_audio']]['noisy_audio'].apply(lambda x: os.path.join(clean,x))
data['clean_audio'] = data['clean_audio'].apply(lambda x: os.path.join(clean,x))

n = len(data) 
train = data.iloc[:int(0.8*n),:]
val = data.iloc[int(0.8*n):,:].reset_index(drop=True)

BATCH_SIZE = 4
train_generator = CustomDataGenerator(df=train,X_col='noisy_audio',y_col='clean_audio',batch_size=BATCH_SIZE)
val_generator = CustomDataGenerator(df=val,X_col='noisy_audio',y_col='clean_audio',batch_size=BATCH_SIZE)

input_img = Input(shape=(1032, 632, 1))

l1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
l2 = Conv2D(64, (3, 3), padding='same', activation='relu')(l1)
l3 = MaxPooling2D(padding='same')(l2)
l4 = Conv2D(128, (3, 3),  padding='same', activation='relu')(l3)
l5 = Conv2D(128, (3, 3), padding='same', activation='relu')(l4)
l6 = MaxPooling2D(padding='same')(l5)
l7 = Conv2D(256, (3, 3), padding='same', activation='relu')(l6)
l8 = UpSampling2D()(l7)
l9 = Conv2D(128, (3, 3), padding='same', activation='relu')(l8)
l10 = Conv2D(128, (3, 3), padding='same', activation='relu')(l9)
l11 = Add()([l5, l10])
l12 = UpSampling2D()(l11)
l13 = Conv2D(64, (3, 3), padding='same', activation='relu')(l12)
l14 = Conv2D(64, (3, 3), padding='same', activation='relu')(l13)
l15 = Add()([l14, l2])
decoded = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(l15)

model = Model(inputs=input_img, outputs=decoded)
print(model.summary())

model.compile(loss='mean_squared_error',optimizer='adam')

model_path = "models/autoencoder.h5"
checkpoint = ModelCheckpoint(model_path,
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 5,
                          verbose = 1,
                          restore_best_weights = True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.3, 
                                            min_lr=0.00000001)

history = model.fit(train_generator,validation_data=val_generator, 
                          epochs=15, callbacks=[earlystop, checkpoint, learning_rate_reduction])

