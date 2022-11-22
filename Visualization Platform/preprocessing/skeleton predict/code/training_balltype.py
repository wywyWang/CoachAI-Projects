import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn import svm
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance
from functions import *
from matplotlib import pyplot as plt
import time

import warnings 
warnings.filterwarnings('ignore')

needed = ['flying_time', 'now_right_x', 'now_right_y', 'now_left_x', 'now_left_y', 
		'next_right_x', 'next_right_y', 'next_left_x', 'next_left_y', 
		'right_delta_x', 'right_delta_y', 'left_delta_x', 'left_delta_y',
		'right_x_speed', 'right_y_speed','right_speed',
		'left_x_speed', 'left_y_speed', 'left_speed', 'hit_height', 'type', 'avg_ball_speed', 
		'hitting_area_number_1', 'hitting_area_number_2', 'hitting_area_number_3', 'hitting_area_number_4', 
		'landing_area_number_1', 'landing_area_number_2', 'landing_area_number_3', 'landing_area_number_4']

train_needed = ['flying_time', 'now_right_x', 'now_right_y', 'now_left_x', 'now_left_y', 
		'next_right_x', 'next_right_y', 'next_left_x', 'next_left_y', 
		'right_delta_x', 'right_delta_y', 'left_delta_x', 'left_delta_y',
		'right_x_speed', 'right_y_speed','right_speed',
		'left_x_speed', 'left_y_speed', 'left_speed', 'avg_ball_speed',  
		'hitting_area_number_1', 'hitting_area_number_2', 'hitting_area_number_3', 'hitting_area_number_4', 
		'landing_area_number_1', 'landing_area_number_2', 'landing_area_number_3', 'landing_area_number_4']
test_needed = ['type']

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
	

	eng_type_to_num = {'cut': 0, 'drive': 1, 'lob': 2, 'long': 3, 'netplay': 4, 'rush': 5, 'smash': 6}

	ball_type = []

	for t in data['type']:
		if ball_type_convertion(t) == 'error':
			print(t)
		ball_type.append(eng_type_to_num[ball_type_convertion(t)])

	data['type'] = ball_type
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

	x_train = data[train_needed+['active', 'passive']]
	y_train = data['type']

	y_train = np.array(y_train).ravel()

	return x_train, y_train

def RandomForest(x_train, y_train, model_name):
	params = {
		'n_estimators': 800,
		'criterion': 'entropy',
		'max_features': 7
	}
	model = RandomForestClassifier(**params)
	model.fit(x_train, y_train)
	joblib.dump(model, model_name)

def SVM(x_train, y_train, model_name):
	model = svm.SVC(kernel='rbf')
	model.fit(x_train, y_train)
	joblib.dump(model, model_name)

def XGBoost(x_train, y_train, model_name):
	params = {
        'learning_rate': 0.01,
        'n_estimators': 800,
        #'max_depth': 8,
        #'min_child_weight': 1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.01,
        'objective':'multi:softmax',
        'scale_pos_weight': 1,
        'num_class': 7
    }
	xgbc = XGBClassifier(**params)
	'''
	grid_params = {
        'learning_rate': [i/100.0 for i in range(1,7)],
        'n_estimators': [700, 800, 900, 1000, 1100, 1200, 1300, 1400],
        #'max_depth': range(1,7),
        #'min_child_weight': range(0,4,1),
        #'subsample': [i/10.0 for i in range(6,10)],
        #'colsample_bytree': [i/10.0 for i in range(6,10)],
        #'gamma': [i/10.0 for i in range(0,10)],
        #'reg_alpha':[0, 1e-5, 1e-3, 1e-2, 0.005, 0.025, 0.01, 0.25, 0.05, 0.10]
    }
	grid = GridSearchCV(xgbc, grid_params, cv = 5)
	
	xgboost_model = grid.fit(x_train, y_train)
	print(xgboost_model.best_params_)
	#{'learning_rate': 0.01, 'max_depth': 8, 'n_estimators': 1100, 'reg_alpha': 0.01}
	'''
	xgbc.fit(x_train, y_train)
	#plot_importance(xgbc)
	#plt.show()
	joblib.dump(xgbc, model_name)

def Run(filename, svm_option, svm_model_name, svm_ball_height_predict_result, xgboost_option, xgboost_model_name, xgboost_ball_height_predict_result, RF_option, RF_model_name, RF_ball_height_predict_result):
	
	if svm_option and svm_model_name != '':
		x_train, y_train = LoadData(filename, svm_ball_height_predict_result)

		print("SVM training...")
		ts = time.time()
		SVM(x_train, y_train, svm_model_name)
		te = time.time()
		print("SVM training done!")
		print("SVM training time: "+str(te-ts))

	if xgboost_option and xgboost_model_name != '':
		x_train, y_train = LoadData(filename, xgboost_ball_height_predict_result)

		print("XGBoost training...")
		ts = time.time()
		XGBoost(x_train, y_train, xgboost_model_name)
		te = time.time()
		print("XGBoost training done!")
		print("XGBoost training time: "+str(te-ts))

	if RF_option and RF_model_name != '':
		x_train, y_train = LoadData(filename, RF_ball_height_predict_result)

		print("Random Forest training...")
		ts = time.time()
		RandomForest(x_train, y_train, RF_model_name)
		te = time.time()
		print("Random Forest training done!")
		print("Random Forest training time: "+str(te-ts))

game_name = "18ENG_TC"

Run('../data/'+str(game_name)+'/'+str(game_name)+'_set1_with_skeleton.csv', \
	False, '../model/'+str(game_name)+'_SVM_balltype.joblib.dat', '../data/'+str(game_name)+'/result/SVM_set1_skeleton_out.csv', \
	True, '../model/'+str(game_name)+'_XGB_balltype.joblib.dat', '../data/'+str(game_name)+'/result/XGB_set1_skeleton_out.csv', \
	True, '../model/'+str(game_name)+'_RF_balltype.joblib.dat', '../data/'+str(game_name)+'/result/RF_set1_skeleton_out.csv')