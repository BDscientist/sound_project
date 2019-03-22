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

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

sequence_length = 40
feature_dimension = 398

def mel_mfcc(part):
    sample_files = open('C:/project2/'+part+'_samples.txt').read().strip().split('\n')
    if part =='train2':
        data_sum = np.zeros((sequence_length,feature_dimension))
        data_squared_sum = np.zeros((sequence_length,feature_dimension))
    if not os.path.exists('C:/project2/'+part+'/spectrum/'):
        os.mkdir('C:/project2/'+part+'/spectrum/')
    for f in sample_files:
        print('%d , %d  >>> %s ' %(sample_files.index(f), len(sample_files),f))
        y,sr = librosa.load('C:/project2/'+part+'/audio/'+f+'.wav')
        mfcc = librosa.feature.mfcc(y=y, sr=22050, S=None, n_mfcc=40, dct_type=2, norm='ortho')
        mfcc_mag , phase = librosa.magphase(mfcc)
        log_normal_mfcc = np.log(1+mfcc_mag*100)
        if part =='train2':
            data_sum +=log_normal_mfcc
            print(data_sum)
            data_squared_sum+=log_normal_mfcc**2
        np.save('C:/project2/'+part+'/spectrum/'+f+'.npy',log_normal_mfcc)
    if part == 'train2': # 모든 파일의 변환이 끝난 후에, 'train'인 경우 평균과 표준편차를 저장합니다.
        data_mean = data_sum / len(sample_files)
        data_std = (data_squared_sum / len(sample_files) - data_mean ** 2) ** 0.5
        numpy.save('data_mean.npy', data_mean)
        numpy.save('data_std.npy', data_std)







if __name__ == '__main__':
  for part in ['train2']:
    mel_mfcc(part)
