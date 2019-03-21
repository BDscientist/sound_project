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



#wav 파일을 로딩하고 mfcc 변환하여 소리 numpy를 추출함
# 파일을 조금 더 전처리하기위해 표준정규화를 만들어 0~1사이에 오게함

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')


y,sr = librosa.load('C:/project1/train2/audio2/ZOOM0021 .wav')

#mel function으로 변환
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
sequence_length = 20
feature_dimension = 398

#print(mfcc.shape)

#print(new_mfcc_numpy.shape)
mfcc_mean = np.mean(mfcc)
mfcc_var = np.var(mfcc)

normal_mfcc = (mfcc - mfcc_mean) / mfcc_var
'''
print(" normal_mfcc > > ",normal_mfcc)
print("\n")
print("normal_mfcc의 형태 > > ",normal_mfcc.shape)

'''
#정규 표준화된 데이터를 로그변환

mfcc_mag , phase = librosa.magphase(normal_mfcc)
log_normal_mfcc = np.log(1+mfcc_mag*1000)

#print(" log_normal_mfcc > > ",log_normal_mfcc)



#sftp 로변환

stft = np.abs(librosa.stft(y,n_fft=1024, hop_length=256))

stft_mean = np.mean(stft)
stft_var = np.var(stft)

normal_stft = (stft - stft_mean) / stft_var
'''
print("normal_stft > > ",normal_stft)
print("\n")
print("normal_stft의 형태",normal_stft.shape)
'''

#sftp 로 변환된 파일을 log화
stft_mag , phase = librosa.magphase(normal_stft)
log_normal_stft = np.log(1+stft_mag*1000)
#print(" log_normal_stft > > ",log_normal_stft)
