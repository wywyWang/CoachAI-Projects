import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.model_selection import *
from xgboost import XGBClassifier, plot_importance
from matplotlib import pyplot as plt
import time

import warnings 
warnings.filterwarnings('ignore')

needed = ['flying_time', 'now_right_x', 'now_right_y', 'now_left_x', 'now_left_y', 
		'next_right_x', 'next_right_y', 'next_left_x', 'next_left_y', 
		'right_delta_x', 'right_delta_y', 'left_delta_x', 'left_delta_y',
		'right_x_speed', 'right_y_speed', 'right_speed',
		'left_x_speed', 'left_y_speed', 'left_speed','hit_height', 'avg_ball_speed', 'type', 
		'hitting_area_number_1', 'hitting_area_number_2', 'hitting_area_number_3', 'hitting_area_number_4', 
		'landing_area_number_1', 'landing_area_number_2', 'landing_area_number_3', 'landing_area_number_4']

train_needed = ['flying_time', 'now_right_x', 'now_right_y', 'now_left_x', 'now_left_y', 
		'next_right_x', 'next_right_y', 'next_left_x', 'next_left_y', 
		'right_delta_x', 'right_delta_y', 'left_delta_x', 'left_delta_y',
		'right_x_speed', 'right_y_speed','right_speed',
		'left_x_speed', 'left_y_speed', 'left_speed', 'avg_ball_speed',
		'hitting_area_number_1', 'hitting_area_number_2', 'hitting_area_number_3', 'hitting_area_number_4', 
		'landing_area_number_1', 'landing_area_number_2', 'landing_area_number_3', 'landing_area_number_4']

test_needed = ['hit_height']

def convert_area(area):
	val = {'E': 0, 'C': 4, 'A': 8, 'B': 12, 'D': 16}
	return float(val[area[0]]+float(area[1]))

def LoadData(filename):
	data = pd.read_csv(filename)
	data = data[needed]
	data.dropna(inplace=True)
	data.reset_index(drop=True, inplace=True)
	data = data[data.type != '未擊球']
	data = data[data.type != '掛網球']
	data = data[data.type != '未過網']
	data = data[data.type != '發球犯規']
	x_train = data[train_needed]
	y_train = data[test_needed].values

	y_train = np.array(y_train).ravel()
	return x_train, y_train

def RandomForest(x_train, y_train, model_name):
	params = {
		'n_estimators': 800,
		'criterion': 'entropy',
		'max_features': 2
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
        #'max_depth': 2000,
        #'min_child_weight': 2,
        'gamma': 0,
        'subsample': 0.83,
        'colsample_bytree': 0.83,
        'reg_alpha': 0.001,
        'objective':'multi:softmax',
        'scale_pos_weight': 1,
        'num_class': 2
    }
	xgbc = XGBClassifier(**params)
	
	'''
	grid_params = {
        'learning_rate': [i/100.0 for i in range(1,6)],
        'n_estimators': [500, 600, 700, 800, 900, 1000, 1100, 1200],
        #'max_depth': range(2,9),
        #'min_child_weight': range(0,10,1),
        #'subsample': [i/10.0 for i in range(6,9)],
        #'colsample_bytree': [i/10.0 for i in range(6,9)],
        #'gamma': [i/10.0 for i in range(0,6)],
        #'reg_alpha':[0, 1e-5, 1e-3, 1e-2, 0.005, 0.025, 0.05, 0.10, 0.15]
    }
	grid = GridSearchCV(xgbc, grid_params, cv = 5)
	
	xgboost_model = grid.fit(x_train, y_train)
	print(xgboost_model.best_params_)
	#{'learning_rate': 0.01, 'max_depth': 2, 'n_estimators': 700, 'reg_alpha': 0.001}
	'''
	
	xgbc.fit(x_train, y_train)
	#plot_importance(xgbc)
	#plt.show()
	joblib.dump(xgbc, model_name)

def Run(filename, svm_option, svm_model_name, xgboost_option, xgboost_model_name, RF_option, RF_model_name):
	x_train, y_train = LoadData(filename)
	if svm_option and svm_model_name != '':
		print("SVM training...")
		ts = time.time()
		SVM(x_train, y_train, svm_model_name)
		te = time.time()
		print("SVM training done!")
		print("SVM training time: "+str(te-ts))
	if xgboost_option and xgboost_model_name != '':
		print("XGBoost training...")
		ts = time.time()
		XGBoost(x_train, y_train, xgboost_model_name)
		te = time.time()
		print("XGBoost training done!")
		print("XGBoost training time: "+str(te-ts))
	if RF_option and RF_model_name != '':
		print("Random Forest training...")
		ts = time.time()
		RandomForest(x_train, y_train, RF_model_name)
		te = time.time()
		print("Random Forest training done!")
		print("Random Forest training time: "+str(te-ts))

game_name = "18ENG_TC"
Run('../data/'+str(game_name)+'/'+str(game_name)+'_set1_with_skeleton.csv', \
	False, '../model/'+str(game_name)+'_SVM_skeleton.joblib.dat', \
	True, '../model/'+str(game_name)+'_XGB_skeleton.joblib.dat', \
	True, '../model/'+str(game_name)+'_RF_skeleton.joblib.dat')