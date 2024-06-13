import glob
import os
import numpy as np
import librosa
import librosa.display    
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def Divide_Wav(y1, sr=44100, k=4):
    sample_list = []
    total_samples = len(y1)
    segment_samples = int((10 / k) * sr)

    for start_sample in range(0, total_samples, segment_samples):
        end_sample = start_sample + segment_samples
        sample_list.append(y1[start_sample:end_sample])
    return sample_list

def Divide_All_Wav(wav_path):
    divide_wav_list = []
    wav_paths = glob.glob(os.path.join(wav_path, "*.wav"))

    for wav_file in wav_paths:
        print(wav_file)
        y1, sr = librosa.load(wav_file, sr=None)
        wav_list = Divide_Wav(y1, sr)
        divide_wav_list.append(wav_list)
    return divide_wav_list

def Make_train_test(divided_list, test_ratio=0.25, sr=44100, n_mfcc=512):
    features = []
    labels = []

    for file_index, segments in enumerate(divided_list):
        print(f"File index: {file_index}, Number of segments: {len(segments)}")  # 디버깅 추가
        for segment in segments:
            if len(segment) == 0:
                continue  # 비어 있는 세그먼트는 무시합니다.
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            print(f"Segment length: {len(segment)}, MFCC mean shape: {mfcc_mean.shape}")  # 디버깅 추가
            features.append(mfcc_mean)
            labels.append(file_index)  # 파일 단위로 레이블을 나누기 위해 사용

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=test_ratio, random_state=0
    )

    print(f"Train features shape: {np.array(train_features).shape}")  # 디버깅 추가
    print(f"Test features shape: {np.array(test_features).shape}")  # 디버깅 추가
    print(f"Train labels: {np.array(train_labels)}")  # 디버깅 추가
    print(f"Test labels: {np.array(test_labels)}")  # 디버깅 추가

    return np.array(train_features), np.array(test_features), np.array(train_labels), np.array(test_labels)

def train_gmm(features, n_components=3, random_state=0):
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(features)
    return gmm

def plot_confusion_matrix(test_labels, predicted_labels, title="Confusion Matrix"):
    cm = confusion_matrix(test_labels, predicted_labels)
    print(cm.shape)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.show()