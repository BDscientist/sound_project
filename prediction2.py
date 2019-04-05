import sys
import os
import io
import ML2
import numpy as np
import prediction
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')


train_data,train_labels = prediction.final_train_dataset, prediction.final_train_labels
test_data,test_labels = prediction.final_test_dataset , prediction.final_test_labels
valid_data,valid_labels = prediction.final_valid_data , prediction.final_valid_labels
