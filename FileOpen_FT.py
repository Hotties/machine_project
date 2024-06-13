from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import math
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

import librosa
import librosa.display


y , sr = librosa.load('C:/Users/82105/Desktop/최종/1.wav', sr=None)

#print(y)
print(len(y))
print('Sampling rate (Hz): %d' %sr)
print('Length (seconds): %.2f' % (len(y) / sr))


plt.figure(figsize =(12,6))
librosa.display.waveshow(y=y,sr=sr)
plt.show()

D = np.abs(librosa.stft(y, n_fft=len(y)-1,hop_length=len(y)))
print(D.shape)

plt.figure(figsize=(12,6))
plt.plot(D)
plt.show()

