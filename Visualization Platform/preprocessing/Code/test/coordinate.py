import xy_to_area as convert
from functions import *
import pandas as pd

hitting = []

def convert_hit_area(filename, savename):
	hit_x = []
	hit_y = []
	hit_area = []
	landing_x = []
	landing_y = []
	landing_area = []
	time = []
	getpoint_player = []
	lose_reason = []
	ball_type = []
	frame = []
	output = pd.DataFrame([]) 

	data = pd.read_csv(filename)
	sets = data['Set']
	rally = data['Rally']
	frame = data['Frame']
	time = data['Time']
	hit_x = data['Y']
	hit_y = data['X']
	getpoint_player = data['Getpoint_player']
	
	for reason in data['Lose_reason']:
		lose_reason.append(map_reason(reason))

	hit_area = convert.to_area(hit_x, hit_y)

	output['set'] = sets
	output['rally'] = rally
	output['frame_num'] = frame
	output['time'] = time
	output['hit_area'] = hit_area
	output['hit_x'] = hit_x
	output['hit_y'] = hit_y
	output['landing_area'] = pd.Series(hit_area[1:])
	output['landing_x'] = pd.Series(hit_x.values[1:])
	output['landing_y'] = pd.Series(hit_y.values[1:])
	output['lose_reason'] = pd.Series(lose_reason)
	output['getpoint_player'] = pd.Series(getpoint_player)
	output['type'] = pd.Series(ball_type)

	output.to_csv(savename, index = False)

def first_hit(filename):
	data = pd.read_csv(filename)
	data = data[['lose_reason', 'getpoint_player']]

	# find who lose first and the reason
	first_blood = ''
	r_cnt = 0
	for i in range(len(data['lose_reason'])):
		if type(data['lose_reason'][i]) == str and type(data['getpoint_player'][i]) == str:
			first_blood = who_first_blood(data['lose_reason'][i], data['getpoint_player'][i])
		if first_blood != '':
			r_cnt = i
			break

	return first_blood, r_cnt

def get_hits(first_blood, cnt, filename):
	start = ''
	if cnt%2:
		start = another_player(first_blood)
	else:
		start = first_blood

	data = pd.read_csv(filename)
	result = pd.DataFrame([])
	

	for i in range(len(data['frame_num'])):
		hitting.append(start)
		start = another_player(start)
		if type(data['lose_reason'][i]) == str and type(data['getpoint_player'][i]) == str:
			start = data['getpoint_player'][i]

	result['hitting'] = hitting
	data['hitting'] = result['hitting']

	data.to_csv(filename, index = False)

def run(filename, savename):
	first_blood = ''
	cnt = 0
	convert_hit_area(filename, savename)

	# repeat to find the player who serve the ball
	
	first_blood, cnt = first_hit(savename)
	get_hits(first_blood, cnt, savename)

def exec(game_names):
	for g in game_names:
		run("../../Data/AccuracyResult/record_segmentation_"+str(g)+".csv", "../../Data/training/data/record_segmentation_"+str(g)+"_out.csv")

exec(["19ASI_CS_10min"])