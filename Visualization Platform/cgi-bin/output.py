import os
import json
import pandas as pd

def export_json(savefile, data):
	with open(savefile, 'w') as outputfile:
		json.dump(json.JSONDecoder().decode(pd.DataFrame(data).to_json(orient='records')), outputfile, separators=(',', ': '), indent = 4)
	
def rally_count(rawfile, predict_file, savefile):
	data = pd.read_csv(rawfile)
	predict_result = pd.read_csv(predict_file)
	needed_data = data[['set', 'rally', 'hit_area', 'getpoint_player', 'lose_reason', 'type']]

	a_score = 0
	b_score = 0
	hit_count = 0
	pre_set = 1
	sets = []
	rally = []
	score = []
	stroke = []
	winner = []

	for i in range(len(needed_data['hit_area'])):
		if type(needed_data['getpoint_player'][i]) != float:
			if needed_data['getpoint_player'][i] == "A":
				a_score += 1
			elif needed_data['getpoint_player'][i] == "B":
				b_score += 1

			sets.append(needed_data['set'][i])
			rally.append(needed_data['rally'][i])
			if needed_data['set'][i] != pre_set:
				pre_set = needed_data['set'][i]
				if a_score > b_score:
					a_score = 1
					b_score = 0
				else:
					a_score = 0
					b_score = 1	

			score.append(str(a_score)+":"+str(b_score))
			stroke.append(hit_count)
			winner.append(needed_data['getpoint_player'][i])


			hit_count = 0

		hit_count += 1
	    
	lose_detail = needed_data[['hit_area','lose_reason']].dropna().reset_index(drop=True)

	cnt = 0
	balltype = []
	for i in range(len(needed_data['getpoint_player'])):
	    if (needed_data['getpoint_player'][i] == 'A' or needed_data['getpoint_player'][i] == 'B') and cnt in range(len(predict_result['prediction'])):
	        balltype.append(predict_result['prediction'][cnt])
	        cnt += 1

	result_data = pd.DataFrame(columns = ["set", "rally", "score", "stroke", "winner", "on_off_court", "balltype", "lose_area"])

	result_data["set"] = sets
	result_data["rally"] = rally
	result_data["score"] = score
	result_data["stroke"] = stroke
	result_data["winner"] = winner
	result_data["on_off_court"] = list(lose_detail['lose_reason'].values)
	result_data["balltype"] = balltype
	result_data["lose_area"] = list(lose_detail['hit_area'].values)

	result_data = (result_data.groupby(['set'], as_index = True)
	            .apply(lambda x: x[['rally','score','stroke','winner','on_off_court','balltype','lose_area']].to_dict('records'))
	            .reset_index()
	            .rename(columns={0:'result'})
	            )

	export_json(savefile, result_data)

def rally_type(rawfile, predict_file, savefile):
	rally_data = pd.read_csv(rawfile)
	hits = pd.read_csv(predict_file)
	result = pd.DataFrame(columns = ["set", "rally", "player", "balltype", "count"])
	ball_id = {'cut': 0, 'drive': 1, 'lob': 2, 'long': 3, 'netplay': 4, 'rush': 5, 'smash': 6}
	sets = 1
	hit_cnt = 0
	score_A = 0
	score_B = 0
	type_A_cnt = [0]*len(ball_id)
	type_B_cnt = [0]*len(ball_id)
	result = pd.DataFrame(columns = ["set", "rally", "player", "balltype", "count"])
	result_A = pd.DataFrame(columns = ["set", "rally", "player", "balltype", "count"])
	result_B = pd.DataFrame(columns = ["set", "rally", "player", "balltype", "count"])

	for i in range(len(rally_data['hitting'])):
		if type(rally_data['getpoint_player'][i]) == str and (rally_data['getpoint_player'][i] == 'A' or rally_data['getpoint_player'][i] == 'B'):
			result_A['balltype'] = list(ball_id.keys())
			result_A['count'] = type_A_cnt
			result_A['set'] = [rally_data['set'][i]]*len(type_A_cnt)
			result_A['rally'] = [rally_data['rally'][i]]*len(type_A_cnt)
			result_A['player'] = ['A']*len(type_A_cnt)

			result_B['balltype'] = list(ball_id.keys())
			result_B['count'] = type_B_cnt
			result_B['set'] = [rally_data['set'][i]]*len(type_B_cnt)
			result_B['rally'] = [rally_data['rally'][i]]*len(type_B_cnt)
			result_B['player'] = ['B']*len(type_B_cnt)

			result = result.append(result_A)
			result = result.append(result_B)
			result_A = pd.DataFrame(columns = ["set", "rally", "player", "balltype", "count"])
			result_B = pd.DataFrame(columns = ["set", "rally", "player", "balltype", "count"])
			type_A_cnt = [0]*len(ball_id)
			type_B_cnt = [0]*len(ball_id)
			continue

		if rally_data['hitting'][i] == 'A':
			type_A_cnt[ball_id[hits['prediction'][hit_cnt]]] += 1
			hit_cnt += 1
		elif rally_data['hitting'][i] == 'B':
			type_B_cnt[ball_id[hits['prediction'][hit_cnt]]] += 1
			hit_cnt += 1

	result = (result.groupby(['set','rally','player'], as_index=False)
            .apply(lambda x: x[['balltype','count']].to_dict('r'))
            .reset_index()
            .rename(columns={0:'result'})
            )
	result = (result.groupby(['set'], as_index=True)
            .apply(lambda x: x[['rally','player','result']].to_dict('r'))
            .reset_index()
            .rename(columns={0:'info'})
            )

	export_json(savefile, result)

def insert_new_game_name(exist_game_file, game_name):
	write = False
	data = []

	if os.path.isfile(exist_game_file):
		with open(exist_game_file) as game:
			data = json.load(game)
			if game_name not in data:
				data.append(game_name)
				write = True
		if write:
			with open(exist_game_file, 'w') as outputfile:
				json.dump(data, outputfile, indent = 4)
	else:
		data.append(game_name)
		with open(exist_game_file, 'w') as outputfile:
			json.dump(data, outputfile, indent = 4)

def run(rawfile, predict_file, rally_count_savefile, rally_type_savefile, exist_game_file, game_name):
	rally_count(rawfile, predict_file, rally_count_savefile)
	rally_type(rawfile, predict_file, rally_type_savefile)
	insert_new_game_name(exist_game_file, game_name)
#run("out.csv", "../../Data/training/result/18IND_TC_predict_result.csv", "rally_count_our.json", "rally_type_our.json", "game_name.json", "NewGameLa")