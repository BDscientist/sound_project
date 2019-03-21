import librosa
import numpy as np
import os
import sys
import io
import librosa.display
from scipy.fftpack import fft
import matplotlib.pylab as plt
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

data,sr= librosa.load('C:/project1/train2/audio2/bass_acoustic_000-024-025.wav') # 저장하는 데이터
data2= np.abs(librosa.stft(data))
#np.save('C:/project1/paper/data2.npy',data2)
