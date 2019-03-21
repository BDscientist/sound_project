import sys
from wav2vec import WavDecoder
from wav2vec.formatter import CSVFormatter
import pickle
import pandas as pd
import wave
import struct

block_size =1024

wave_data = wave.open('C:\cleanfile\sample.wav')
nframes = wave_data.getnframes()
sample_count = 0

data = wave_data.readframes(block_size)
while data !='':
    num_words = len(data) /2
    fmt_str = '<%dh' %num_words

    numbers = struct.unpack(fmt_str,data)
    for sample in numbers:
        print("%f, %f\n" % (sample_count, sample))
        sample_count +=1

    data = wave_data.readframes(block_size)






'''


decoder = WavDecoder('C:\cleanfile\sample.wav')


formatter = CSVFormatter(decoder)


df = pd.read_csv(formatter)


print(df)
'''
