import util1

# 오디오 파일을 분할하고 특징을 추출
wav_path = 'C:/Users/82105/Desktop/최종'
divided_list = util1.Divide_All_Wav(wav_path)

# 학습 및 테스트 데이터 생성
train_features, test_features, train_labels, test_labels = util1.Make_train_test(divided_list)
# GMM 모델 학습
gmm = util1.train_gmm(train_features)

# 테스트 데이터에 대한 예측
predicted_labels = gmm.predict(test_features)
print(train_labels.shape,predicted_labels.shape)

# 혼동행렬 시각화
util1.plot_confusion_matrix(test_labels, predicted_labels)