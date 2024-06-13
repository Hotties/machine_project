import glob
import os
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import math
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

import librosa
import librosa.display    

def Divide_Wav(y1,sr=44100,k = 4):
    sample_list = np.array([])
    total_samples = len(y1)

    segment_samples = (10/k)*sr

    segment_samples = int(segment_samples)

    for start_sample in range(0,total_samples,segment_samples):
        end_sample = start_sample + segment_samples
        sample_list.append(y1[start_sample:end_sample])
    return sample_list



def Divide_All_Wav(wav_path):

    divide_wav_list = np.array([])

    wav_paths = glob.glob(os.path.join(wav_path,"*.wav"))
    for i in enumerate (wav_path):
        y1,sr = librosa.load(i,sr=None)
        wav_list = Divide_Wav(y1,sr)
        divide_wav_list.append(wav_list)
    return divide_wav_list

def Make_train_test(list):
    
    test_wav = np.array[()]
    train_wav = np.array[()]
    train_mfcc = np.array[()]
    for cnt,i in range(0,4),enumerate(list):
        if(np.array_equal(i,list[cnt])):
            mfcc = librosa.feature.mfcc(i,sr=44100,n_mfcc=512)
            np.append(test_wav,mfcc)            
        else:
            np.repeat(train_mfcc,i,axis=0)
        mfcc1 = librosa.feature.mfcc(train_mfcc,sr=44100)
        np.append(train_wav,mfcc1)

    return test_wav,train_wav