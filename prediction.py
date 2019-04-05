import sys
import os
import io
import ML2
import ML_TEST
import numpy as np
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

frame =[]
dataset = ML_TEST.prepare_data()
for i in range(1000):
    final_test_dataset , final_test_labels = ML_TEST.get_random_sample('test')
    trans_dataset = np.transpose(final_test_dataset)
    frame.append(final_test_labels)
    np.savetxt("c:/project2/testdata/"+str(i)+".csv",final_test_dataset,delimiter=",")

np.savetxt("c:/project2/testdata/labels.csv",frame,delimiter=",")
    #print(final_dataset.shape() , data_labels.shape() )

#print(frame)


'''
    def newdata():
        dataset = ML2.prepare_data()
        for i in range(1,10000):
            final_dataset, data_labels = ML2.get_random_sample('train')

        return final_dataset,data_labels

        '''
