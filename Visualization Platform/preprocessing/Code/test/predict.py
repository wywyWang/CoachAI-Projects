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

def feature_importance(xgbc):
    plot_importance(xgbc)
    plt.savefig('featureimp.png')
    plt.clf()
    plt.close()

def plot_cm(cm, groundtruth, grid_predictions, classes):

    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xlabel('Actual Class')
    plt.ylabel('Predicted Class')
    plt.xticks(np.arange(len(classes)-1), classes)
    plt.yticks(np.arange(len(classes)-1), classes)
   
    for j, i in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i == j:
            plt.text(i+0.03, j+0.2, str(format(cm[j, i], 'd'))+'\n'+str( round(precision_score(groundtruth, grid_predictions, average=None)[i]*100,1))+'%', 
            color="white" if cm[j, i] > cm.max()/2. else "black", 
            horizontalalignment="center")
        else:
            plt.text(i, j, format(cm[j, i], 'd'), 
            color="white" if cm[j, i] > cm.max()/2. else "black", 
            horizontalalignment="center")

    plt.savefig('confusion_matrix.png')
    plt.close(0)

def confusion_matrix_chart(groundtruth, grid_predictions, type_labels):
    cm = confusion_matrix(groundtruth, grid_predictions)
    print('Confusion matrix with the best estimator:', '\r', cm)
    print('Precision for each label:', precision_score(groundtruth, grid_predictions, average=None))
    print('Recall for each label:', recall_score(groundtruth, grid_predictions, average=None))
    print('Total Accuracy: ', accuracy_score(groundtruth, grid_predictions))
    plot_cm(cm, groundtruth, grid_predictions, type_labels)
    plt.clf()
    plt.close()

def radar_chart(new_dict, type_labels, groundtruth, grid_predictions):
    plt.rcParams['axes.unicode_minus'] = False

    yv = [0, 0, 0, 0, 0, 0, 0]
    pred = [0, 0, 0, 0, 0, 0, 0]
    total = 0

    type_list = [new_dict[i] for i in groundtruth]
    type_set = set(type_list)
    for i in type_set:
        yv[i] = type_list.count(i)
        total += yv[i]

    type_list2 = [new_dict[i] for i in grid_predictions]
    type_set2 = set(type_list2)
    for i in type_set2:
        pred[i] = type_list2.count(i)

    N = len(yv)
    angles = np.linspace(0, 2*np.pi, N, endpoint = False)
    yv = np.concatenate((yv, [yv[0]]))
    pred = np.concatenate((pred, [pred[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(0)
    ax = fig.add_subplot(111, polar = True)
    data_radar_real = np.concatenate((yv, [yv[0]]))
    ax.plot(angles, yv, linewidth=2, label = 'real')
    data_radar_predict = np.concatenate((pred, [pred[0]]))
    ax.plot(angles, pred, linewidth=2, label = 'predict')
    pred = np.asarray(pred)
    propotion = np.round(pred / total*100,2)
    propotion = propotion.astype(str)
    ax.set_thetagrids(angles * 180 / np.pi, type_labels + '\n' + propotion + '%')
    plt.title('type predition',loc ='right')
    plt.legend(bbox_to_anchor = (1.1, 0), bbox_transform = ax.transAxes)
    plt.savefig('radar.png')
    plt.clf()
    plt.close()

def result_chart(xgbc, groundtruth, grid_predictions, type_labels, new_dict):
    # feature importance
    feature_importance(xgbc)
    # confusion matrix
    confusion_matrix_chart(groundtruth, grid_predictions, type_labels)
    # radar chart
    radar_chart(new_dict, type_labels, groundtruth, grid_predictions)


def run(filename_predict, model_path, filename_result):
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
    prediction = xgboost_model.predict(x_predict)

    # output
    result = pd.DataFrame([])
    result['prediction'] = prediction
    result['answer'] = data_predict[:,-1]
    result.to_csv(filename_result,index=None)

    # plot graph
    #result_chart(xgboost_model, data_predict[:, -1], prediction, type_labels, new_dict)

    # print precision and recall
    print("Accuracy: "+str(accuracy_score(data_predict[:, -1], prediction)))
    print("Precision: "+str(precision_score(data_predict[:, -1], prediction, labels = ['cut', 'drive', 'lob', 'long', 'netplay', 'rush', 'smash'], average='macro')))
    print("Recall: "+str(recall_score(data_predict[:, -1], prediction, labels = ['cut', 'drive', 'lob', 'long', 'netplay', 'rush', 'smash'], average='macro')))

def verify(pre_dir, filename_predict, model_path, result_dir, filename_result):
    
    # pre_dir: where the files after preprocessing saved
    # filename_predict: the file that we want to predict
    # model_path: the model generated after running the training code
    # result_dir: the directory that store the result
    # filename_result: where the result will be saved

    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    print("Start predict...")
    run(filename_predict, model_path, filename_result)
    print("Prediction done...")

def exec(game_names):
    for g in game_names:
        verify("./", "../../Data/training/data/"+str(g)+"_preprocessed.csv", "../../Data/training/model/model.joblib.dat", "../../Data/training/result", "../../Data/training/result/"+str(g)+"_result.csv")

exec(["19ASI_CS_10min"])