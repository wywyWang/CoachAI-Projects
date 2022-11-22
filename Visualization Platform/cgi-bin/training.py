import csv
import itertools
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.externals import joblib
from sklearn.metrics import *
from sklearn.model_selection import *
import warnings 
import os
warnings.filterwarnings('ignore')

def train(filename_train, model_path):

    # load dataset
    data_train = np.array([])
    with open(filename_train, newline='', encoding='utf8') as f:
        reader = csv.reader(f)
        next(reader, None)
        c = 0
        for row in reader:
            if c == 0:
                data_train = np.hstack((data_train, np.array(row)))
                c = 1
            else:
                data_train = np.vstack((data_train, np.array(row)))


    # sperate feature and target
    x_train = data_train[:,:-1] # feature
    y_train = data_train[:,-1] # target

    params = {
        'learning_rate': 0.01,
        'n_estimators': 500,
        'max_depth': 4,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective':'binary:logistic',
        'scale_pos_weight': 1
    }

    xgbc = xgb.XGBClassifier(**params)

    grid_params = {
        #'learning_rate': [i/1000.0 for i in range(1,2)],
        #'max_depth': range(3,6),
        #'min_child_weight': range(0,10,1),
        #'subsample': [i/10.0 for i in range(6,9)],
        #'colsample_bytree': [i/10.0 for i in range(6,9)],
        #'gamma': [i/10.0 for i in range(0,5)],
        'reg_alpha':[1, 2, 5, 10]
    }

    # training
    grid = GridSearchCV(xgbc, grid_params)
    xgboost_model = grid.fit(x_train, y_train, eval_metric='auc')
    xgbc.fit(x_train, y_train)

    # save model
    joblib.dump(xgbc, model_path)

def verify(pre_dir, filename_train, model_path):

    # pre_dir: where the files after preprocessing saved
    # filename_train: the file that we want to use for training
    # model_path: the model generated after running the training code

    if os.path.isdir(pre_dir) and os.path.isfile(filename_train) and not os.path.isfile(model_path):
        
        print("Start training...")
        train(filename_train, model_path)
        print("Training done...")

    else:
        if os.path.isfile(model_path):
            print("Already exist model also named: "+str(model_path))
        if not os.path.isdir(pre_dir):
            print("No such directory named: "+str(pre_dir))
        if not os.path.isfile(filename_train):
            print("No such file named: "+str(filename_train))