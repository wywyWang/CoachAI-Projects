import pandas as pd
import numpy as np
import math
import xy_to_area as mapping

def find_index(data, target):
	for idx in range(len(data)):
		if data[idx] == target:
			return idx

def get_hitting_pos(set_info, skeleton_info, top_is_Taiwan):
	hitting_pos = []
	times = []
	pre_id = 0
	for idx in set_info.index:
		skeleton_idx = find_index(skeleton_info['frame'], set_info['frame_num'][idx])
		pos = []
		#print(idx)
		#print(set_info['time'][idx])
		if skeleton_idx == None:
			pos.append('')
			pos.append('')
			pos.append('')
			pos.append('')
			times.append('')
		else:
			if top_is_Taiwan:
				if set_info['player'][idx] == 'A':
					pos.append(skeleton_info['top_right_x'][skeleton_idx])
					pos.append(skeleton_info['top_right_y'][skeleton_idx])
					pos.append(skeleton_info['top_left_x'][skeleton_idx])
					pos.append(skeleton_info['top_left_y'][skeleton_idx])
					try:
						#times.append(int(set_info['time'][idx].split(':')[0])*60*60*1000+int(set_info['time'][idx].split(':')[1])*60*1000+int(set_info['time'][idx].split(':')[2].split('.')[0])*1000+int(set_info['time'][idx].split(':')[2].split('.')[1])/100*100)
						times.append(int(set_info['time'][idx].split(':')[0])*60*60*1000+int(set_info['time'][idx].split(':')[1])*60*1000+int(set_info['time'][idx].split(':')[2].split('.')[0])*1000)
					except:
						#print(set_info['time'][idx].split(':'))
						#print(int(set_info['time'][idx].split(':')[0])*60*60*1000+int(set_info['time'][idx].split(':')[1])*60*1000+int(set_info['time'][idx].split(':')[2])*1000)
						times.append(int(set_info['time'][idx].split(':')[0])*60*60*1000+int(set_info['time'][idx].split(':')[1])*60*1000+int(set_info['time'][idx].split(':')[2])*1000)
				else:
					pos.append(skeleton_info['bot_right_x'][skeleton_idx])
					pos.append(skeleton_info['bot_right_y'][skeleton_idx])
					pos.append(skeleton_info['bot_left_x'][skeleton_idx])
					pos.append(skeleton_info['bot_left_y'][skeleton_idx])
					try:
						#print(set_info['time'][idx].split(':'))
						#print(int(set_info['time'][idx].split(':')[0])*60*60*1000+int(set_info['time'][idx].split(':')[1])*60*1000+int(set_info['time'][idx].split(':')[2].split('.')[0])*1000+int(set_info['time'][idx].split(':')[2].split('.')[1])/100*100)
						#times.append(int(set_info['time'][idx].split(':')[0])*60*60*1000+int(set_info['time'][idx].split(':')[1])*60*1000+int(set_info['time'][idx].split(':')[2].split('.')[0])*1000+int(set_info['time'][idx].split(':')[2].split('.')[1])/100*100)
						times.append(int(set_info['time'][idx].split(':')[0])*60*60*1000+int(set_info['time'][idx].split(':')[1])*60*1000+int(set_info['time'][idx].split(':')[2].split('.')[0])*1000)

					except:
						#print(set_info['time'][idx].split(':'))
						#print(int(set_info['time'][idx].split(':')[0])*60*60*1000+int(set_info['time'][idx].split(':')[1])*60*1000+int(set_info['time'][idx].split(':')[2])*1000)
						times.append(int(set_info['time'][idx].split(':')[0])*60*60*1000+int(set_info['time'][idx].split(':')[1])*60*1000+int(set_info['time'][idx].split(':')[2])*1000)
			else:
				if set_info['player'][idx] == 'A':
					pos.append(skeleton_info['bot_right_x'][skeleton_idx])
					pos.append(skeleton_info['bot_right_y'][skeleton_idx])
					pos.append(skeleton_info['bot_left_x'][skeleton_idx])
					pos.append(skeleton_info['bot_left_y'][skeleton_idx])
					try:
						times.append(int(set_info['time'][idx].split(':')[0])*60*60*1000+int(set_info['time'][idx].split(':')[1])*60*1000+int(set_info['time'][idx].split(':')[2].split('.')[0])*1000)
						#print(set_info['time'][idx].split(':'))
						#print(int(set_info['time'][idx].split(':')[0])*60*60*1000+int(set_info['time'][idx].split(':')[1])*60*1000+int(set_info['time'][idx].split(':')[2].split('.')[0])*1000+int(set_info['time'][idx].split(':')[2].split('.')[1])/100*100)
						#times.append(int(set_info['time'][idx].split(':')[0])*60*60*1000+int(set_info['time'][idx].split(':')[1])*60*1000+int(set_info['time'][idx].split(':')[2].split('.')[0])*1000+int(set_info['time'][idx].split(':')[2].split('.')[1])/100*100)
					except:
						#print(set_info['time'][idx].split(':'))
						#print(int(set_info['time'][idx].split(':')[0])*60*60*1000+int(set_info['time'][idx].split(':')[1])*60*1000+int(set_info['time'][idx].split(':')[2])*1000)
						times.append(int(set_info['time'][idx].split(':')[0])*60*60*1000+int(set_info['time'][idx].split(':')[1])*60*1000+int(set_info['time'][idx].split(':')[2])*1000)
				else:
					pos.append(skeleton_info['top_right_x'][skeleton_idx])
					pos.append(skeleton_info['top_right_y'][skeleton_idx])
					pos.append(skeleton_info['top_left_x'][skeleton_idx])
					pos.append(skeleton_info['top_left_y'][skeleton_idx])
					try:
						times.append(int(set_info['time'][idx].split(':')[0])*60*60*1000+int(set_info['time'][idx].split(':')[1])*60*1000+int(set_info['time'][idx].split(':')[2].split('.')[0])*1000)
						#print(set_info['time'][idx].split(':'))
						#print(int(set_info['time'][idx].split(':')[0])*60*60*1000+int(set_info['time'][idx].split(':')[1])*60*1000+int(set_info['time'][idx].split(':')[2].split('.')[0])*1000+int(set_info['time'][idx].split(':')[2].split('.')[1])/100*100)
						#times.append(int(set_info['time'][idx].split(':')[0])*60*60*1000+int(set_info['time'][idx].split(':')[1])*60*1000+int(set_info['time'][idx].split(':')[2].split('.')[0])*1000+int(set_info['time'][idx].split(':')[2].split('.')[1])/100*100)
					except:
						#print(set_info['time'][idx].split(':'))
						#print(int(set_info['time'][idx].split(':')[0])*60*60*1000+int(set_info['time'][idx].split(':')[1])*60*1000+int(set_info['time'][idx].split(':')[2])*1000)
						times.append(int(set_info['time'][idx].split(':')[0])*60*60*1000+int(set_info['time'][idx].split(':')[1])*60*1000+int(set_info['time'][idx].split(':')[2])*1000)
				
		hitting_pos.append(pos)

	return hitting_pos, times

def pos_test(hitting_pos, now):
	start_pos = tuple()
	end_pos = tuple()
	start_idx = 0
	end_idx = len(hitting_pos)-1

	if now > 0:
		start_idx = now
	if now+2 < len(hitting_pos):
		end_idx = now+2

	if hitting_pos[start_idx, 0] == '' or hitting_pos[start_idx, 1] == '' or hitting_pos[start_idx, 2] == '' or hitting_pos[start_idx, 3] == '':
		return (False, 0, 0, 0, 0)
	start_pos = (hitting_pos[start_idx, 0], hitting_pos[start_idx, 1], hitting_pos[start_idx, 2], hitting_pos[start_idx, 3])
	
	if hitting_pos[end_idx, 0] == '' or hitting_pos[end_idx, 1] == '' or hitting_pos[end_idx, 2] == '' or hitting_pos[end_idx, 3] == '':
		return (False, 0, 0, 0, 0)
	end_pos = (hitting_pos[end_idx, 0], hitting_pos[end_idx, 1], hitting_pos[end_idx, 2], hitting_pos[end_idx, 3])

	return (True, int(start_pos[0])-int(end_pos[0]), int(start_pos[1])-int(end_pos[1]), int(start_pos[2])-int(end_pos[2]), int(start_pos[3])-int(end_pos[3]))


def Merge(set_num, total_set, setinfo, skeleton_file, top_is_Taiwan, savename, change_side):
	set_info = pd.read_csv(setinfo)
	skeleton_info = pd.read_csv(skeleton_file)
	sec_per_frame = 0.04
	right_delta_x = []
	right_delta_y = []
	left_delta_x = []
	left_delta_y = []

	right_x_speed = []
	right_y_speed = []
	left_x_speed = []
	left_y_speed = []
	
	right_speed = []
	left_speed = []

	avg_ball_speed = []
	times = []
	hitting_pos = []
	left_right_distance = []
	flying_time = []

	if set_num != total_set:
		hitting_pos = np.array(get_hitting_pos(set_info, skeleton_info, top_is_Taiwan)[0])
		times = np.array(get_hitting_pos(set_info, skeleton_info, top_is_Taiwan)[1])
	else:
		first = pd.DataFrame([])
		second = pd.DataFrame([])
		break_point = 0

		for i in range(len(set_info['roundscore_A'])):
			if int(set_info['roundscore_A'][i]) == 12 or int(set_info['roundscore_B'][i]) == 12:
				first = set_info[:][:i]
				second = set_info[:][i:]
				break_point = i
				print(i)
				break
		#print(first)
		first.reset_index()
		#print(first)
		second.reset_index()
		#print(len(first))
		#print(len(second))
		first_part, first_time = get_hitting_pos(first, skeleton_info, top_is_Taiwan)
		second_part, second_time = get_hitting_pos(second, skeleton_info, top_is_Taiwan)
		
		if change_side:
			hitting_pos = np.array(second_part)
			times = np.array(second_time)
			set_info = set_info[:][break_point:]
			set_info = set_info.reset_index(drop=True)
			print(len(hitting_pos))
		else:
			hitting_pos = np.array(first_part)
			times = np.array(first_time)
			set_info = set_info[:][:break_point]
			set_info = set_info.reset_index(drop=True)
			print(len(hitting_pos))

		#hitting_pos = np.array(first_part+second_part)
		#times = np.array(first_time+second_time)
	set_info = set_info.reset_index(drop=True)

	for i in range(len(hitting_pos)-2):
		r_delta_x = int()
		r_delta_y = int()
		l_delta_x = int()
		l_delta_y = int()

		if hitting_pos[i, 0] != '' and hitting_pos[i, 1] != '' and hitting_pos[i, 2] != '' and hitting_pos[i, 3] != '':
			now_delta_x = float(hitting_pos[i, 0])-float(hitting_pos[i, 2])
			now_delta_x = now_delta_x*now_delta_x
			now_delta_y = float(hitting_pos[i, 1])-float(hitting_pos[i, 3])
			now_delta_y = now_delta_y*now_delta_y
			left_right_distance.append(math.sqrt(now_delta_x+now_delta_y))

		if hitting_pos[i, 0] != '' and hitting_pos[i+2, 0] != '':
			r_delta_x = float(hitting_pos[i+2, 0])-float(hitting_pos[i, 0])
			right_delta_x.append((r_delta_x%10)*10)

		if hitting_pos[i, 1] != '' and hitting_pos[i+2, 1] != '':
			r_delta_y = float(hitting_pos[i+2, 1])-float(hitting_pos[i, 1])
			right_delta_y.append((r_delta_y%10)*10)

		if hitting_pos[i, 2] != '' and hitting_pos[i+2, 2] != '':
			l_delta_x = float(hitting_pos[i+2, 2])-float(hitting_pos[i, 2])
			left_delta_x.append((l_delta_x%10)*10)

		if hitting_pos[i, 3] != '' and hitting_pos[i+2, 3] != '':
			l_delta_y = float(hitting_pos[i+2, 3])-float(hitting_pos[i, 3])
			left_delta_y.append((l_delta_y%10)*10)

		x_right = -1
		x_left = -1
		y_right = -1
		y_left = -1
		right_ok = False
		left_ok = False

		if hitting_pos[i, 0] != '' and hitting_pos[i+2, 0] != '' and hitting_pos[i, 1] != '' and hitting_pos[i+2, 1] != '':
			right_ok = True
			delta_x = abs(r_delta_x)
			delta_y = abs(r_delta_y)
			x_right = delta_x
			y_right = delta_y
			right_x_speed.append((delta_x/100)/sec_per_frame)
			right_y_speed.append((delta_y/100)/sec_per_frame)
			right_speed.append((math.sqrt(delta_x*delta_x+delta_y*delta_y)/100)/sec_per_frame)

		if hitting_pos[i, 2] != '' and hitting_pos[i+2, 2] != '' and hitting_pos[i, 3] != '' and hitting_pos[i+2, 3] != '':
			left_ok = True
			delta_x = abs(l_delta_x)
			delta_y = abs(l_delta_y)
			x_left = delta_x
			y_left = delta_y
			left_x_speed.append((delta_x/100)/sec_per_frame)
			left_y_speed.append((delta_y/100)/sec_per_frame)
			left_speed.append((math.sqrt(delta_x*delta_x+delta_y*delta_y)/100)/sec_per_frame)
		
		if pos_test(hitting_pos, i)[0] and times[i+2] != '' and times[i] != '':
			right_x_delta = pos_test(hitting_pos, i)[1]
			right_y_delta = pos_test(hitting_pos, i)[2]
			left_x_delta = pos_test(hitting_pos, i)[3]
			left_y_delta = pos_test(hitting_pos, i)[4]

			dx = ((right_x_delta+left_x_delta)/2)**2
			dy = ((right_y_delta+left_y_delta)/2)**2
			dt = float(times[i+2])-float(times[i]) 

			try:
				avg_ball_speed.append(abs(math.sqrt(dx+dy)/dt))
			except:
				avg_ball_speed.append('')
		else:
			avg_ball_speed.append('')
      
	hitting_area_number_1 = []
	hitting_area_number_2 = []
	hitting_area_number_3 = []
	hitting_area_number_4 = []
	
	for i in range (0, len(hitting_pos[0:, 1])):
		if hitting_pos[:, 1][i] == '' or hitting_pos[:, 3][i] == '':
			hitting_area_number_1.append('')
			hitting_area_number_2.append('')
			hitting_area_number_3.append('')
			hitting_area_number_4.append('')
		else:
			mid_y = (int(hitting_pos[:, 1][i])+int(hitting_pos[:, 3][i]))/2
			if mid_y < 0 or mid_y >= 935:
				#hitting_area_number.append(4)
				hitting_area_number_4.append(1)
				hitting_area_number_3.append(0)
				hitting_area_number_2.append(0)
				hitting_area_number_1.append(0)

			elif (mid_y >= 0 and mid_y < 74) or (mid_y >= 861 and mid_y < 935):
				#hitting_area_number.append(3)
				hitting_area_number_4.append(0)
				hitting_area_number_3.append(1)
				hitting_area_number_2.append(0)
				hitting_area_number_1.append(0)
			elif (mid_y >= 74 and mid_y < 307) or (mid_y >= 629 and mid_y < 861):
				#hitting_area_number.append(2)
				hitting_area_number_4.append(0)
				hitting_area_number_3.append(0)
				hitting_area_number_2.append(1)
				hitting_area_number_1.append(0)
			elif mid_y >= 307 and mid_y < 629:
				#hitting_area_number.append(1)
				hitting_area_number_4.append(0)
				hitting_area_number_3.append(0)
				hitting_area_number_2.append(0)
				hitting_area_number_1.append(1)
	
	'''
	for i in range (0, len(hitting_pos[1:, 1])):
		if hitting_pos[1:, 1][i]=='' or hitting_pos[1:, 3][i]=='':
			landing_area_number.append('')
		else:
			mid_y = (int(hitting_pos[1:, 1][i])+int(hitting_pos[1:, 3][i]))/2
			if mid_y < 0 :
				landing_area_number.append(4)
			elif mid_y >= 0 and mid_y < 74:
				landing_area_number.append(3)
			elif mid_y >= 74 and mid_y < 307:
				landing_area_number.append(2)
			elif mid_y >= 307 and mid_y < 629:
				landing_area_number.append(1)
			elif mid_y >= 629 and mid_y < 861:
				landing_area_number.append(2)
			elif mid_y >= 861 and mid_y < 935:
				landing_area_number.append(3)
			elif mid_y >= 935 :
				landing_area_number.append(4)
	'''
	landing_area_number_1 = hitting_area_number_1[1:]
	landing_area_number_2 = hitting_area_number_2[1:]
	landing_area_number_3 = hitting_area_number_3[1:]
	landing_area_number_4 = hitting_area_number_4[1:]
	landing_area_number_1.append('')
	landing_area_number_2.append('')
	landing_area_number_3.append('')
	landing_area_number_4.append('')

	for i in range(len(times)-1):
		if times[i] != '' and times[i+1] != '':
			flying_time.append(float(times[i+1])-float(times[i])) 
		else:
			flying_time.append('')
	#print(flying_time)
	set_info['flying_time'] = pd.Series(list(flying_time))

	set_info['now_right_x'] = pd.Series(list(hitting_pos[:, 0]))
	set_info['now_right_y'] = pd.Series(list(hitting_pos[:, 1]))

	set_info['now_left_x'] = pd.Series(list(hitting_pos[:, 2]))
	set_info['now_left_y'] = pd.Series(list(hitting_pos[:, 3]))

	set_info['next_right_x'] = pd.Series(list(hitting_pos[1:, 0]))
	set_info['next_right_y'] = pd.Series(list(hitting_pos[1:, 1]))

	set_info['next_left_x'] = pd.Series(list(hitting_pos[1:, 2]))
	set_info['next_left_y'] = pd.Series(list(hitting_pos[1:, 3]))

	set_info['right_delta_x'] = pd.Series(right_delta_x)
	set_info['right_delta_y'] = pd.Series(right_delta_y)

	set_info['left_delta_x'] = pd.Series(left_delta_x)
	set_info['left_delta_y'] = pd.Series(left_delta_y)

	set_info['right_x_speed'] = pd.Series(right_x_speed)
	set_info['right_y_speed'] = pd.Series(right_y_speed)
	set_info['right_speed'] = pd.Series(right_speed)

	set_info['left_x_speed'] = pd.Series(left_x_speed)
	set_info['left_y_speed'] = pd.Series(left_y_speed)
	set_info['left_speed'] = pd.Series(left_speed)

	set_info['avg_ball_speed'] = pd.Series(avg_ball_speed)
	set_info['left_right_distance'] = pd.Series(left_right_distance)

	set_info['hitting_area_number_1'] = pd.Series(hitting_area_number_1)
	set_info['hitting_area_number_2'] = pd.Series(hitting_area_number_2)
	set_info['hitting_area_number_3'] = pd.Series(hitting_area_number_3)
	set_info['hitting_area_number_4'] = pd.Series(hitting_area_number_4)

	set_info['landing_area_number_1'] = pd.Series(landing_area_number_1)
	set_info['landing_area_number_2'] = pd.Series(landing_area_number_2)
	set_info['landing_area_number_3'] = pd.Series(landing_area_number_3)
	set_info['landing_area_number_4'] = pd.Series(landing_area_number_4)

	set_info.to_csv(savename, index=False, encoding = 'utf-8')

def run(set_num, total_set, game_name, top_is_Taiwan, change_side):
	if change_side:
		Merge(set_num, total_set, '../data/'+str(game_name)+'/set'+str(set_num)+'.csv', '../data/'+str(game_name)+'/player_skeleton/'+str(game_name)+'_set'+str(set_num)+'_skeleton.csv', top_is_Taiwan, '../data/'+str(game_name)+'/'+str(game_name)+'_set'+str(set_num)+'-1_with_skeleton.csv', change_side)
	else:
		Merge(set_num, total_set, '../data/'+str(game_name)+'/set'+str(set_num)+'.csv', '../data/'+str(game_name)+'/player_skeleton/'+str(game_name)+'_set'+str(set_num)+'_skeleton.csv', top_is_Taiwan, '../data/'+str(game_name)+'/'+str(game_name)+'_set'+str(set_num)+'_with_skeleton.csv', change_side)

def exec(number_of_sets):
	top_Taiwan = False #18IND_TC: True # 18ENG_TC: False
	change_side = False
	game_name = "18ENG_TC"

	for i in range(1, number_of_sets+1):
		print("Start merging set "+str(i))
		run(i, number_of_sets, game_name, top_Taiwan, change_side)
		print("Set "+str(i)+" merge done")
		top_Taiwan = not top_Taiwan
	
	if number_of_sets == 3:
		change_side = True
		run(number_of_sets, 3, game_name, top_Taiwan, change_side)

exec(3)