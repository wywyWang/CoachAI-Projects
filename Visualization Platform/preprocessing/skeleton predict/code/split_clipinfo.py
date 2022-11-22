import pandas as pd

needed = ['unique_id', 'rally', 'ball_round', 'time', 'frame_num', 'player', 'server', 'type', 'around_head', 'backhand', 
			'hit_area', 'hit_x', 'hit_y', 'hit_height', 'lose_reason', 'win_reason', 'roundscore_A', 'roundscore_B',
			'getpoint_player', 'landing_area', 'landing_x', 'landing_y', 'landing_height']
save = ['rally', 'ball_round', 'time', 'frame_num', 'player', 'server', 'type', 'around_head', 'backhand', 
			'hit_area', 'hit_x', 'hit_y', 'hit_height', 'lose_reason', 'win_reason', 'roundscore_A', 'roundscore_B',
			'getpoint_player', 'landing_area', 'landing_x', 'landing_y', 'landing_height']

pre_dir = "../data/"

def load_data(filename):
	data = pd.read_excel(filename)
	data = data[needed]
	return data

def split_data(filename, game_name):
	needed_data = load_data(filename)
	pre_idx = 0
	now_id = needed_data['unique_id'][0]

	for i in range(len(needed_data)):
		if needed_data['unique_id'][i] != now_id:
			set_data = needed_data[pre_idx:i]
			set_data.reset_index(drop=True, inplace=True)
			set_data.to_csv(pre_dir+game_name+"_set"+str(now_id.split("-")[-1])+".csv", index=False, encoding = 'utf-8')
			pre_idx = i
			now_id = needed_data['unique_id'][i]

	set_data = needed_data[pre_idx:len(needed_data)]
	set_data.reset_index(drop=True, inplace=True)
	set_data.to_csv(pre_dir+game_name+"_set"+str(now_id.split("-")[-1])+".csv", index=False, encoding = 'utf-8')

def run(filenames):
	for f in filenames:
		split_data(pre_dir+"clip_info_"+f+".xlsx", f)

run(["18IND_TC"])