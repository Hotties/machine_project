from sklearn.mixture import GaussianMixture
from util import *

def gaussian(list):
    gmm  = GaussianMixture(n_components=3,random_state=42)
    gmm.fit_predict(list)
    return gmm

if __name__ == "__main__":
    
    ###파일 불러와서 divide_wav_list 만들기

    divide_wav_list = Divide_All_Wav('path')

    ##그 후 mfcc추출하기
    for i in enumerate(divide_wav_list):##10번
        test_wav,train_wav = Make_train_test(divide_wav_list)
    ##그 다음 gmm 실행
        for j,k in enumerate(test_wav,train_wav):##4번
            gmm,gmm_labels = gaussian(k)

            
        
    ##혼동행렬 생성
    
    ## 정확도 계산