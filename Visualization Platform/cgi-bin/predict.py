import csv
import pandas as pd
import numpy as np
import xgboost as xgb
import itertools

import math
import warnings 
import os
from sklearn.externals import joblib
from sklearn.metrics import *
from xgboost import plot_importance
import matplotlib.pyplot as plt
from matplotlib import pyplot

warnings.filterwarnings('ignore')

def exec(filename_predict, model_path, filename_result):
    label_name_dict = {
        0:"cut",
        1:"drive",
        2:"lob",
        3:"long",
        4:"netplay",
        5:"rush",
        6:"smash",
        7:""
    }
    new_dict = {v : k for k, v in label_name_dict.items()}

    type_labels = pd.DataFrame(label_name_dict, index=[0])
    type_labels = type_labels.values[0]

    xgboost_model = joblib.load(model_path)
    
    # load dataset
    data_predict = np.array([])

    with open(filename_predict, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        c = 0
        for row in reader:
            if c == 0:
                data_predict = np.hstack((data_predict, np.array(row)))
                c = 1
            else:
                data_predict = np.vstack((data_predict, np.array(row)))

    x_predict = data_predict[:,:-1]

    # prediction
    grid_predictions = xgboost_model.predict(x_predict)

    # output
    pd.DataFrame(grid_predictions,columns=['prediction']).to_csv(filename_result,index=None)

    # print precision and recall
    #print("Precision: "+str(precision_score(data_predict[:, -1], grid_predictions, labels = ['cut', 'drive', 'lob', 'long', 'netplay', 'rush', 'smash'], average=None)))
    #print("Recall: "+str(recall_score(data_predict[:, -1], grid_predictions, labels = ['cut', 'drive', 'lob', 'long', 'netplay', 'rush', 'smash'], average=None)))

def verify(pre_dir, filename_predict, model_path, result_dir, filename_result):
    
    # pre_dir: where the files after preprocessing saved
    # filename_predict: the file that we want to predict
    # model_path: the model generated after running the training code
    # result_dir: the directory that store the result
    # filename_result: where the result will be saved

    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    print("Start predict...")
    print("<br>")
    exec(filename_predict, model_path, filename_result)
    print("Prediction done...")
    print("<br>")

#verify("./", "set1_after.csv", "../preprocessing/Data/training/model/model.joblib.dat", "./rrr", "./rrr/resulttttt.csv")