import sys
import os
import io
import ML2
import numpy as np
import prediction
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

pieces= []
for i in range(1000):
    path = 'c:/project2/testdata/%d.csv' %i
    frame2 = pd.read_csv(path)

    pieces.append(frame2)

names = pd.concat(pieces,axis=1,ignore_index=True)
test_data = names.T

np.savetxt("c:/project2/test_data.csv",test_data,delimiter=",")
