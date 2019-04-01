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
import prediction
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')



def rmsle(y,y_,convertExp=True):
    if convertExp:
        y= numpy.exp(y),
        y_ = numpy.exp(y_)
    log1 = numpy.nan_to_num(numpy.array([numpy.log(v+1)for v in y]))
    log2 = numpy.nan_to_num(numpy.array([numpy.log(v+1)for v in y_]))
    calc = (log1 - log2)**2
    return numpy.sqrt(numpy.mean(calc))




datasets , datalabels = prediction.final_dataset, prediction.data_labels

model_xgb = xgb.XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                        min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0,
                        objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

model_xgb.fit(datasets,datalabels)
xgb_train_pred  = model_xgb.predict(datasets)
xgb_pred = numpy.expm1(model_xgb.predict(datalabels))
print(rmsle(datalabels, xgb_train_pred))
