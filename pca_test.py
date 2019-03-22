import librosa
import numpy as np
import os
import sys
import io
import librosa
import librosa.display
import IPython.display
from librosa.feature import melspectrogram
import numpy as np
from scipy.fftpack import fft
import matplotlib.pylab as plt


y,sr = librosa.load('C:/project1/train/audio/ZOOM0021 .wav',sr = 160000)
stft = librosa.stft(y,n_fft=1024, hop_length=256).T


#sftp 로 변환된 파일을 log화
stft_mag , phase = librosa.magphase(stft)
log_stft = np.log(1+stft_mag*1000)
print(" log_normal_stft > > ",log_stft)
print(log_stft.shape)

#covariance 변환
feature = log_stft
cov_matrix = np.cov(feature)
print(cov_matrix)


#eigen value 값 구하기
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
print("Eigenvectors \n%s" %eig_vecs)
print("eig value \n" , eig_vecs.shape)

# 각 컴포넌트의 실제 값 들 임의로 구하기
test_matrix=[]
for i in range(0,100):
        test_matrix.append(eig_vals[i] / sum(eig_vals))

print(test_matrix)
