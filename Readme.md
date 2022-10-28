# Audio Enhancement and Denoising using Autoencoders

## Audio Enhancement
Audio Enhancement services are the scientific process of clarifying audio recordings using non-destructive techniques to preserve speech quality. This is crucial for the trier of fact can make determinations about the events within the recorded evidence. Enhancement techniques are applied to remove sounds like static, furnace and air conditioning fans, hums and other distracting sounds. Therefore, these distracting noises are also known as "unwanted sounds". Then, once we remove the unwanted sounds, we can apply enhancement processes. These enhancement processes will increase the "wanted sound" like speech. 

## Audio Denoising
Audio Denoising is the process of removing noises from a speech without affecting the quality of the speech. Here, the noises are any unwanted audio segments for the human hearing like vehicle horn sounds, wind noise, or even static noise.

![](https://cdn.analyticsvidhya.com/wp-content/uploads/2021/07/traditional-noise-filtering-850x528.png)

## Dataset 
https://www.kaggle.com/datasets/quadeer15sh/noisyandcleanaudios

## Autoencoder Architecture

![](https://editor.analyticsvidhya.com/uploads/98612autoencoder.JPG)

## Noisy Audio to Clean Audio
- The architecture below gives a brief overview of how an audio singal is converted into its time a frequency domain correspondence in the form of a spectrogram using Short Term Fourier Transform (STFT) and is then passed on to an autoencoder which reconstructs the enhanced and denoised spectrogram. The reconstructed Spectrogram is then converted back into the audio signal form using Inverse Short Term Fourier Transform (ISTFT) (Griffin-Lim algorithm). 

**Note:** The architecture shown below is of a variational autoencoder, however the one used for the reconstruction resembles a U-Net like architecture.

![](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/mvae-ass/img/CVAEsourcemodel.bmp)

## Python Libraries Required
```
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install tensorflow
pip install tqdm
pip install librosa
pip install opencv-python
pip install soundfile
```



