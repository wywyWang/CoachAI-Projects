import csv
import pandas as pd
import numpy as np
import xgboost as xgb
import itertools
import functions

from sklearn.externals import joblib
from sklearn.metrics import *
from xgboost import plot_importance
import matplotlib.pyplot as plt
from functions import *

import warnings 
warnings.filterwarnings('ignore')

needed = ['flying_time', 'now_right_x', 'now_right_y', 'now_left_x', 'now_left_y', 
		'next_right_x', 'next_right_y', 'next_left_x', 'next_left_y', 
		'right_delta_x', 'right_delta_y', 'left_delta_x', 'left_delta_y',
		'right_x_speed', 'right_y_speed','right_speed',
		'left_x_speed', 'left_y_speed', 'left_speed', 'hit_height', 'type', 'avg_ball_speed',  
        'hitting_area_number_1', 'hitting_area_number_2', 'hitting_area_number_3', 'hitting_area_number_4', 
        'landing_area_number_1', 'landing_area_number_2', 'landing_area_number_3', 'landing_area_number_4']

test_needed = ['flying_time', 'now_right_x', 'now_right_y', 'now_left_x', 'now_left_y', 
		'next_right_x', 'next_right_y', 'next_left_x', 'next_left_y', 
		'right_delta_x', 'right_delta_y', 'left_delta_x', 'left_delta_y',
		'right_x_speed', 'right_y_speed','right_speed',
		'left_x_speed', 'left_y_speed', 'left_speed', 'avg_ball_speed',   
        'hitting_area_number_1', 'hitting_area_number_2', 'hitting_area_number_3', 'hitting_area_number_4', 
        'landing_area_number_1', 'landing_area_number_2', 'landing_area_number_3', 'landing_area_number_4']

def LoadData(filename, ball_height_predict):
    data = pd.read_csv(filename)
    ball_height = pd.read_csv(ball_height_predict)
    data = data[needed]
    data.dropna(inplace=True)
    data = data[data.type != '未擊球']
    data = data[data.type != '掛網球']
    data = data[data.type != '未過網']
    data = data[data.type != '發球犯規']
    data.reset_index(drop=True, inplace=True)

    active = []
    passive = []

    for i in ball_height['Predict']:
        if i == 1:
            active.append(1)
            passive.append(0)
        else:
            active.append(0)
            passive.append(1)

    data['active'] = active
    data['passive'] = passive
    
    x_predict = data[test_needed+['active', 'passive']]

    return x_predict

def plot_Confusion_Matrix(game_name, set_now, model_type, cm, groundtruth, grid_predictions, classes, change_side):
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xlabel('Actual Class')
    plt.ylabel('Predicted Class')
    plt.xticks(np.arange(len(classes)), classes)
    plt.yticks(np.arange(len(classes)), classes)

    for j, i in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i == j:
            plt.text(i+0.03, j+0.2, str(format(cm[j, i], 'd'))+'\n'+str( round(precision_score(groundtruth, grid_predictions, labels=classes, average=None)[i]*100,1))+'%', 
            color="white" if cm[j, i] > cm.max()/2. else "black", 
            horizontalalignment="center")
        else:
            plt.text(i, j, format(cm[j, i], 'd'), 
            color="white" if cm[j, i] > cm.max()/2. else "black", 
            horizontalalignment="center")

    if change_side:
        plt.savefig('../data/'+str(game_name)+'/img/'+str(model_type)+'-1_set'+str(set_now)+'_balltype_confusion_matrix.png')
    else:
        plt.savefig('../data/'+str(game_name)+'/img/'+str(model_type)+'_set'+str(set_now)+'_balltype_confusion_matrix.png')

    plt.close(0)

def plot_chart(game_name, set_now, model_type, model, groundtruth, grid_predictions, labels, change_side):
    # confusion matrix
    plot_Confusion_Matrix(set_now, model_type, confusion_matrix(groundtruth, grid_predictions, labels=labels), groundtruth, grid_predictions, labels, change_side)
    plt.clf()
    plt.close()

def RandomForest(filename, x_predict, model_name, RF_outputname, set_now, game_name, change_side):
    data = pd.read_csv(filename)
    data = data[needed]
    data.dropna(inplace=True)
    data = data[data.type != '未擊球']
    data = data[data.type != '掛網球']
    data = data[data.type != '未過網']
    data = data[data.type != '發球犯規']
    data.reset_index(drop=True, inplace=True)

    label = [0, 1, 2, 3, 4, 5, 6]
    type_to_num = {'cut': 0, 'drive': 1, 'lob': 2, 'long': 3, 'netplay': 4, 'rush': 5, 'smash': 6}
    num_to_type = {0: 'cut', 1: 'drive', 2: 'lob', 3: 'long', 4: 'netplay', 5: 'rush', 6: 'smash'}
    real_num = []
    real_eng = []
    predict_result_eng = []

    model = joblib.load(model_name)
    prediction = model.predict(x_predict)
    
    for ball in data['type']:
        real_eng.append(num_to_type[type_to_num[ball_type_convertion(ball)]])
        real_num.append(type_to_num[ball_type_convertion(ball)])

    for ball in prediction:
        predict_result_eng.append(num_to_type[ball])

    result = pd.DataFrame([])
    result['Real'] = real_num
    result['Predict'] = prediction

    result.to_csv(RF_outputname,index=None)

    cnt = 0
    for i in range(len(result['Real'])):
        if result['Real'][i] == result['Predict'][i]:
            cnt+=1
    
    print("RF Total correct: "+str(cnt))
    print("RF Total number: "+str(len(prediction)))
    print("RF Accuracy: "+str(accuracy_score(real_num, prediction)))
    print("RF Overall precision: "+str(precision_score(real_num, prediction, labels = label, average='micro')))
    print("RF Overall recall: "+str(recall_score(real_num, prediction, labels = label, average='micro')))

    #print("RF Average precision: "+str(precision_score(real_num, prediction, labels = label, average=None)))
    #print("RF Average recall: "+str(recall_score(real_num, prediction, labels = label, average=None)))


def SVM(filename, x_predict, model_name, svm_outputname, set_now, game_name, change_side):
    data = pd.read_csv(filename)
    data = data[needed]
    data.dropna(inplace=True)
    data = data[data.type != '未擊球']
    data = data[data.type != '掛網球']
    data = data[data.type != '未過網']
    data = data[data.type != '發球犯規']
    data.reset_index(drop=True, inplace=True)

    label = [0, 1, 2, 3, 4, 5, 6]
    type_to_num = {'cut': 0, 'drive': 1, 'lob': 2, 'long': 3, 'netplay': 4, 'rush': 5, 'smash': 6}
    num_to_type = {0: 'cut', 1: 'drive', 2: 'lob', 3: 'long', 4: 'netplay', 5: 'rush', 6: 'smash'}
    real_num = []
    real_eng = []
    predict_result_eng = []

    model = joblib.load(model_name)
    prediction = model.predict(x_predict)
    
    for ball in data['type']:
        try:
            real_eng.append(num_to_type[type_to_num[ball_type_convertion(ball)]])
            real_num.append(type_to_num[ball_type_convertion(ball)])
        except:
            print(ball)

    for ball in prediction:
        predict_result_eng.append(num_to_type[ball])

    result = pd.DataFrame([])
    result['Real'] = real_num
    result['Predict'] = prediction

    result.to_csv(svm_outputname,index=None)
    
    cnt = 0
    for i in range(len(result['Real'])):
        if result['Real'][i] == result['Predict'][i]:
            cnt+=1
    
    print("SVM Total correct: "+str(cnt))
    print("SVM Total number: "+str(len(prediction)))
    print("SVM Accuracy: "+str(accuracy_score(real_num, prediction)))
    print("SVM Overall precision: "+str(precision_score(real_num, prediction, labels = label, average='micro')))
    print("SVM Overall recall: "+str(recall_score(real_num, prediction, labels = label, average='micro')))

    #print("SVM Average precision: "+str(precision_score(real_num, prediction, labels = label, average=None)))
    #print("SVM Average recall: "+str(recall_score(real_num, prediction, labels = label, average=None)))


def XGBoost(filename, x_predict, xgboost_model_name, xgboost_outputname, set_now, game_name, change_side):
    data = pd.read_csv(filename)
    data = data[needed]
    data.dropna(inplace=True)
    data = data[data.type != '未擊球']
    data = data[data.type != '掛網球']
    data = data[data.type != '未過網']
    data = data[data.type != '發球犯規']
    data.reset_index(drop=True, inplace=True)

    label = [0, 1, 2, 3, 4, 5, 6]
    type_to_num = {'cut': 0, 'drive': 1, 'lob': 2, 'long': 3, 'netplay': 4, 'rush': 5, 'smash': 6}
    num_to_type = {0: 'cut', 1: 'drive', 2: 'lob', 3: 'long', 4: 'netplay', 5: 'rush', 6: 'smash'}
    real_num = []
    real_eng = []
    predict_result_eng = []

    xgboost_model = joblib.load(xgboost_model_name)
    prediction = xgboost_model.predict(x_predict)

    for ball in data['type']:
    	real_eng.append(num_to_type[type_to_num[ball_type_convertion(ball)]])
    	real_num.append(type_to_num[ball_type_convertion(ball)])

    for ball in prediction:
    	predict_result_eng.append(num_to_type[ball])

    # output
    result = pd.DataFrame([])
    result['Real'] = real_num
    result['Predict'] = prediction

    result.to_csv(xgboost_outputname, index=None)

    # print precision and recall
    cnt = 0
    for i in range(len(result['Real'])):
        if result['Real'][i] == result['Predict'][i]:
            cnt+=1
    
    print("XGBoost Total correct: "+str(cnt))
    print("XGBoost Total number: "+str(len(prediction)))
    print("XGBoost Accuracy: "+str(accuracy_score(real_num, prediction)))
    print("XGBoost Average precision: "+str(precision_score(real_num, prediction, labels = label, average='micro')))
    print("XGBoost Average recall: "+str(recall_score(real_num, prediction, labels = label, average='micro')))

    #print("XGBoost Overall precision: "+str(precision_score(real_num, prediction, labels = label, average=None)))
    #print("XGBoost Overall Recall: "+str(recall_score(real_num, prediction, labels = label, average=None)))
    
    # plot result chart
    #plot_chart(game_name, set_now, "XGB", xgboost_model, real_eng, predict_result_eng, list(type_to_num.keys()), change_side)

def Run(game_name, change_side, set_now, filename, svm_option, svm_model_name, svm_prediction_result_file, svm_outputname, xgboost_option, xgboost_model_name, xgboost_prediction_result_file, xgboost_outputname, RF_option, RF_model_name, RF_prediction_result_file, RF_outputname):
	
    if svm_option and svm_model_name != '':
        if change_side:
            print("SVM predicting set"+str(set_now)+"-1...")
        else:
            print("SVM predicting set"+str(set_now)+"...")
        x_predict = LoadData(filename, svm_prediction_result_file)
        SVM(filename, x_predict, svm_model_name, svm_outputname, set_now, game_name, change_side)
        #print("SVM predict set"+str(set_now)+" done!")
        print("---------------------------------------------------")
    if xgboost_option and xgboost_model_name != '':
        if change_side:
            print("XGBoost predicting set"+str(set_now)+"-1...")
        else:
            print("XGBoost predicting set"+str(set_now)+"...")
        x_predict = LoadData(filename, xgboost_prediction_result_file)
        XGBoost(filename, x_predict, xgboost_model_name, xgboost_outputname, set_now, game_name, change_side)
        #print("XGBoost predict set"+str(set_now)+" done!")
        print("---------------------------------------------------")
    if RF_option and RF_model_name != '':
        if change_side:
            print("Random Forest predicting set"+str(set_now)+"-1...")
        else:
            print("Random Forest predicting set"+str(set_now)+"...")
        RandomForest(filename, x_predict, RF_model_name, RF_outputname, set_now, game_name, change_side)
        #print("Random Forest predict set"+str(set_now)+" done!")
        print("---------------------------------------------------")

def exec(predict_set):
    change_side = False
    game_name = "18ENG_TC"

    for i in predict_set:
        Run(game_name, change_side, i ,'../data/'+str(game_name)+'/'+str(game_name)+'_set'+str(i)+'_with_skeleton.csv', \
            False, '../model/'+str(game_name)+'_SVM_balltype.joblib.dat', '../data/'+str(game_name)+'/result/SVM_set'+str(i)+'_skeleton_out.csv', '../data/'+str(game_name)+'/result/SVM_set'+str(i)+'_balltype_out.csv', \
            True, '../model/'+str(game_name)+'_XGB_balltype.joblib.dat', '../data/'+str(game_name)+'/result/XGB_set'+str(i)+'_skeleton_out.csv', '../data/'+str(game_name)+'/result/XGB_set'+str(i)+'_balltype_out.csv', \
            True, '../model/'+str(game_name)+'_RF_balltype.joblib.dat', '../data/'+str(game_name)+'/result/RF_set'+str(i)+'_skeleton_out.csv', '../data/'+str(game_name)+'/result/RF_set'+str(i)+'_balltype_out.csv')

    '''
    change_side = True
    
    Run(change_side, 3 ,'../data/set'+str(i)+'-1_with_skeleton.csv', True, '../model/SVM_balltype.joblib.dat', '../data/result/SVM_set'+str(i)+'-1_skeleton_out.csv', '../data/result/SVM_set'+str(i)+'-1_balltype_out.csv', \
            True, '../model/XGB_balltype.joblib.dat', '../data/result/XGB_set'+str(i)+'-1_skeleton_out.csv', '../data/result/XGB_set'+str(i)+'-1_balltype_out.csv', \
            True, '../model/RF_balltype.joblib.dat', '../data/result/RF_set'+str(i)+'-1_skeleton_out.csv', '../data/result/RF_set'+str(i)+'-1_balltype_out.csv')
    '''
exec([1, 2, 3])	