from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet,Lasso,BayesianRidge,LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings
import sys
import os
import io
import ML2
import numpy
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

dataset = ML2.prepare_data()
for i in range(1,100):
    final_dataset, data_labels = ML2.get_random_sample('train')
    print(final_dataset.shape() , data_labels.shape() )


'''
    def newdata():
        dataset = ML2.prepare_data()
        for i in range(1,10000):
            final_dataset, data_labels = ML2.get_random_sample('train')

        return final_dataset,data_labels

        '''
