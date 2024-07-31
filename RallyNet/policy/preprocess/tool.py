import pickle
import torch
import random
import numpy as np
import torch.nn.functional as F

from bisect import bisect_left
from operator import itemgetter
from copy import copy, deepcopy

K, EXP_NUM = None, None
STATE_LEN, ACTION_LEN = None, None

op_trainning_trajectories = None
op_testing_trajectories = None
experience_trajectories = None
Type_Dict_Load = None
Play_Dict_Load = None
Ball_Dict_Load = None
Oppo_Dict_Load = None

ope_dataset_names = None

def tool_global(mt, en, sl, al, ope):
    global K
    global EXP_NUM
    global STATE_LEN
    global ACTION_LEN


    global ope_dataset_names
    global op_trainning_trajectories
    global op_testing_trajectories
    global experience_trajectories
    global Type_Dict_Load
    global Play_Dict_Load
    global Ball_Dict_Load
    global Oppo_Dict_Load

    K, EXP_NUM = mt, en

    STATE_LEN, ACTION_LEN = sl, al

    ope_dataset_names = ope

    op_trainning_trajectories = pickle.load(open('Data/'+ str(K) + '/' + ope_dataset_names[0], 'rb'))
    op_testing_trajectories = pickle.load(open('Data/'+ str(K) + '/' + ope_dataset_names[1], 'rb'))
    experience_trajectories = pickle.load(open('Data/'+ str(K) + '/' + ope_dataset_names[2], 'rb'))
    
    Type_Dict_Load = pickle.load(open('Data/'+ str(K) + '/experience/Type_Dict.pkl', 'rb'))
    Play_Dict_Load = pickle.load(open('Data/'+ str(K) + '/experience/Play_Dict.pkl', 'rb'))
    Ball_Dict_Load = pickle.load(open('Data/'+ str(K) + '/experience/Ball_Dict.pkl', 'rb'))
    Oppo_Dict_Load = pickle.load(open('Data/'+ str(K) + '/experience/Oppo_Dict.pkl', 'rb'))

target_players = ['PUSARLA V. Sindhu', 'NG Ka Long Angus', 'Pornpawee CHOCHUWONG', 'Neslihan YIGIT', 'Viktor AXELSEN', 'LIEW Daren', 'Jonatan CHRISTIE', 'LEE Zii Jia', 'Anders ANTONSEN', 'Carolina MARIN', 'Kento MOMOTA', 'CHEN Yufei', 'Hans-Kristian Solberg VITTINGHUS', 'WANG Tzu Wei', 'LEE Cheuk Yiu', 'An Se Young', 'Rasmus GEMKE', 'Evgeniya KOSETSKAYA', 'Sameer VERMA', 'Khosit PHETPRADAB', 'KIDAMBI Srikanth', 'TAI Tzu Ying', 'Supanida KATETHONG', 'Busanan ONGBAMRUNGPHAN', 'CHEN Long', 'Ratchanok INTANON', 'CHOU Tien Chen', 'Michelle LI', 'Anthony Sinisuka GINTING', 'Mia BLICHFELDT', 'SHI Yuqi']
match_state = ['rally', 'ball_round', 'player_score', 'opponent_score', 'score_status']
player_state = ['player_location_x', 'player_location_y']
ball_state = ['ball_distance','ball_distance_x','ball_distance_y',  'hit_x', 'hit_y']
opponent_state = ['opponent_type', 'opponent_location_x', 'opponent_location_y', 'opponent_move_x', 'opponent_move_y']
state_col = match_state + player_state + ball_state + opponent_state
action_col = ['landing_x', 'landing_y', 'player_type','player_move_location_x','player_move_location_y']
state_action_dividing = len(state_col)

def state_action_separation(trajectories):
    state = []
    action = []
    for traj in trajectories:
        temp_state_traj = []
        temp_action_traj = []
        for step in traj:
            temp_state_traj.append(step[:STATE_LEN])
            temp_action_traj.append(step[-ACTION_LEN:])
        state.append(temp_state_traj)
        action.append(temp_action_traj)
    return state, action

def pad_sequences_states(seqs, seq_lengths, state_length=STATE_LEN):
    seq_tensor = torch.zeros((len(seqs), seq_lengths.max(), state_length)).float()
    for idx, (seq, seqlen) in enumerate(zip(seqs, seq_lengths)):
        seq_tensor[idx, :seqlen, :state_length] = torch.FloatTensor(seq).view(-1, state_length)
    return seq_tensor

def pad_sequences_actions(seqs, seq_lengths, action_length=ACTION_LEN):
    seq_tensor = torch.zeros((len(seqs), seq_lengths.max(), action_length)).long()
    for idx, (seq, seqlen) in enumerate(zip(seqs, seq_lengths)):
        seq_tensor[idx, :seqlen,  :action_length] = torch.LongTensor(seq).view(-1, action_length)
    return seq_tensor

def del_tensor_ele(arr, index):
    arr1 = arr[:,:,:index]
    arr2 = arr[:,:,index+1:]
    return torch.cat((arr1,arr2),dim=2)

def get_tensor_ele(arr, index):
    arr1 = arr[:,:,index]
    return arr1

def coord2area(point_x, point_y):
    mistake_landing_area = 33

    point_x = (point_x * (355/2)) + (355/2)
    point_y = (point_y * 240) + 240

    area1 = [[50,150],[104,204],1]
    area2 = [[104,150],[177.5,204],2]
    area3 = [[177.5,150],[251,204],3]
    area4 = [[251,150],[305,204],4]
    row1 = [area1, area2, area3, area4]

    area5 = [[50,204],[104,258],5]
    area6 = [[104,204],[177.5,258],6]
    area7 = [[177.5,204],[251,258],7]
    area8 = [[251,204],[305,258],8]
    row2 = [area5, area6, area7, area8]

    area9 = [[50,258],[104,312],9]
    area10 = [[104,258],[177.5,312],10]
    area11 = [[177.5,258],[251,312],11]
    area12 = [[251,258],[305,312],12]
    row3 = [area9, area10, area11, area12]
    
    area13 = [[50,312],[104, 366],13]
    area14 = [[104,312],[177.5,366],14]
    area15 = [[177.5,312],[251,366],15]
    area16 = [[251,312],[305,366],16]
    row4 = [area13, area14, area15, area16]

    area17 = [[50,366],[104,423],17]
    area18 = [[104,366],[177.5,423],18]
    area19 = [[177.5,366],[251,423],19]
    area20 = [[251,366],[305,423],20]
    row5 = [area17, area18, area19, area20]

    area21 = [[50,423],[104,480],21]
    area22 = [[104,423],[177.5,480],22]
    area23 = [[177.5,423],[251,480],23]
    area24 = [[251,423],[305,480],24]
    row6 = [area21, area22, area23, area24]

    area25 = [[305,366],[355,480],25]
    area26 = [[305,204],[355,366],26]
    area27 = [[305,0],[355,204],27]
    area28 = [[177.5,0],[305,150],28]
    row7 = [area25, area26, area27, area28]

    area29 = [[0,366],[50,480],29]
    area30 = [[0,204],[50,366],30]
    area31 = [[0,0],[50,204],31]
    area32 = [[50,0],[177.5,150],32]
    row8 = [area29, area30, area31, area32]

    check_area_list = row1 + row2 + row3 + row4 + row5 + row6 + row7 + row8
    hit_area = mistake_landing_area
    for check_area in check_area_list:
        if point_x >= check_area[0][0] and point_y >= check_area[0][1] and point_x <= check_area[1][0] and point_y <= check_area[1][1]:
            hit_area = check_area[2]
    return hit_area

def list_subtraction(p1,p2):
    point1 = p1.copy()
    point2 = p2.copy()
    v = list(map(lambda x: x[0]-x[1], zip(point1, point2)))
    return v[0], v[1]

def opponent_defeated(player_current_state, target_number):
    player_current_state[1] = player_current_state[1] + 1
    for i in range(len(player_current_state)):
        if i == 17:
            player_current_state[i] = target_number
        else:
            player_current_state[i] = 0
    return player_current_state


def check_tatic_next_status(player_action): # mistake --> True
    land_error = False
    if coord2area(player_action[0], player_action[1]) > 24:
        return True
    if ((land_error == True) or (player_action[0] == 0 and player_action[1] == 1) or (player_action[2] == 11)):
        return True
    return False

def state_transformer(player_action, player_current_state, target_number): # Calculate the opponent's state based on state and action
    # game over
    game_over = 0
    for state in player_current_state[:len(state_col)]:
        if int(state) == 0:
            game_over += 1
    
    # game over，No matter what the agent does, the state passed to the opponent will be the end of the game
    if game_over == len(state_col):
        player_current_state = opponent_defeated(player_current_state, target_number)
        return player_current_state
    
    hit_error = False
    
    # Opponent makes a mistake, no matter what the agent does, the opponent receives the status Game Over
    if coord2area(player_current_state[10], player_current_state[11]) > 24:
        hit_error = True
    if ((hit_error == True) or (player_current_state[10] == 0 and player_current_state[11] == 1) or (player_current_state[12] == 11)):
        player_current_state = opponent_defeated(player_current_state, target_number)
        return player_current_state

    land_error = False
    
    if coord2area(player_action[0], player_action[1]) > 24:
        land_error = True

    if ((land_error == True) or (player_action[0] == 0 and player_action[1] == 1) or (player_action[2] == 11)):
        player_current_state = opponent_defeated(player_current_state, target_number)
        return player_current_state

    return_state_list = player_current_state.copy()
    opponent_score = return_state_list[2]
    player_score = return_state_list[3]
    score_status = player_score - opponent_score
    return_state_list[2] = player_score
    return_state_list[3] = opponent_score
    return_state_list[4] = score_status
    return_state_list[1] = return_state_list[1] + 1 # ball round +1
    hit_x = player_action[0]
    hit_y = player_action[1]
    opponent_type = player_action[2]
    
    opponent_location_x = player_action[3]
    opponent_location_y = player_action[4]

    return_state_list[5] = player_current_state[13]
    return_state_list[6] = player_current_state[14]
    return_state_list[7] = (return_state_list[8]**2 + return_state_list[9]**2)**0.5
    return_state_list[8], return_state_list[9] = list_subtraction([player_action[0], player_action[1]],[player_current_state[13],player_current_state[14]])
    return_state_list[10] = hit_x
    return_state_list[11] = hit_y
    return_state_list[12] = opponent_type
    return_state_list[13] = opponent_location_x
    return_state_list[14] = opponent_location_y
    return_state_list[15], return_state_list[16] = list_subtraction([player_action[3], player_action[4]],[player_current_state[5],player_current_state[6]])
    return_state_list[17] = target_number
    return return_state_list

def sample_inputs(length, states_actions, f_or_h, no_touch):
    raw_length = length
    exchange = False
    tmp_states_actions = copy(states_actions)
    if f_or_h == 'h':
        tmp_states_actions = np.delete(tmp_states_actions, no_touch, 0)
        tmp_states_actions = tmp_states_actions.tolist()
    
    if len(tmp_states_actions) < length:
        if len(tmp_states_actions) == 0:
            tmp_states_actions = random.sample(states_actions, 1)
            print("DONE add one", length)
        else:
            length = len(tmp_states_actions)
            print("DONE one epoch", length)
    
    output = random.sample(tmp_states_actions, length)

    record = []
    for i in range(len(output)):
        record.append(states_actions.index(output[i]))
    
    output_states = []
    output_actions = []
    op_output_states = []
    op_output_actions = []

    for i in range(len(output)):
        temp_states = []
        temp_actions = []
        op_temp_states = []
        op_temp_actions = []

        search_trajectory = []
        for j in range(len(output[i])):
            temp_states.append(output[i][j][:STATE_LEN])
            temp_actions.append(output[i][j][-ACTION_LEN:])
            search_trajectory.append(output[i][j])
        
        if search_trajectory[0][1] == 1:
            exchange = False
        else:
            exchange = True
        
        if f_or_h == 'h':
            op_traj = op_testing_trajectories[record[i]]
        else:
            op_traj = op_trainning_trajectories[record[i]]

        for j in range(len(op_traj)):
            op_temp_states.append(op_traj[j][:STATE_LEN])
            op_temp_actions.append(op_traj[j][-ACTION_LEN:])

        tensor_states = torch.from_numpy(np.asarray(temp_states)).type(torch.FloatTensor)
        tensor_actions = torch.from_numpy(np.asarray(temp_actions)).type(torch.FloatTensor)
        if exchange == True:
            op_output_states.append(tensor_states)
            op_output_actions.append(tensor_actions)
        else:
            output_states.append(tensor_states)
            output_actions.append(tensor_actions)

        op_tensor_states = torch.from_numpy(np.asarray(op_temp_states)).type(torch.FloatTensor)
        op_tensor_actions = torch.from_numpy(np.asarray(op_temp_actions)).type(torch.FloatTensor)
        if exchange == True:
            output_states.append(op_tensor_states)
            output_actions.append(op_tensor_actions)
        else:
            op_output_states.append(op_tensor_states)
            op_output_actions.append(op_tensor_actions)

    if len(output_states) != raw_length:
        for i in range(raw_length - len(output_states)):
            output_states.append(output_states[0].clone())
            output_actions.append(output_actions[0].clone())
            op_output_states.append(op_output_states[0].clone())
            op_output_actions.append(op_output_actions[0].clone())
    
    return output_states, output_actions, op_output_states, op_output_actions, record

def get_keys_from_dict(dictionary, keys, have_keys):
    try:
        intersection = list(set.intersection(*map(set, [keys, have_keys])))
        out = itemgetter(*intersection)(dictionary)
        if type(out) == list:
            tmp = [out]
            out = tuple(tmp)
        out = list(set.union(*map(set, out)))
    except:
        intersection = random.sample(have_keys, int(len(have_keys)/5))
        out = itemgetter(*intersection)(dictionary)
        if type(out) == list:
            tmp = [out]
            out = tuple(tmp)
        out = list(set.union(*map(set, out)))
    return out

def mapping_get_sim_traj_i(now_area_loc):
    search_list = [ Type_Dict_Load.get(now_area_loc[0]), Play_Dict_Load.get(now_area_loc[1]), Ball_Dict_Load.get(now_area_loc[2]), Oppo_Dict_Load.get(now_area_loc[3])]
    t_cover_num = 1
    a_cover_num = 1
    b_cover_num = 1
    c_cover_num = 1
    while None in search_list:
        extend_cover_index_list = [i for i,x in enumerate(search_list) if x == None]
        
        if 0 in extend_cover_index_list :
            all_type_location = get_other_keys(now_area_loc[0], a_cover_num)
            search_list[0] = get_keys_from_dict(Type_Dict_Load, all_type_location, list(Type_Dict_Load.keys()))
            t_cover_num += 1
        
        if 1 in extend_cover_index_list :
            all_player_location = get_other_keys(now_area_loc[1], a_cover_num)
            search_list[1] = get_keys_from_dict(Play_Dict_Load, all_player_location, list(Play_Dict_Load.keys()))
            a_cover_num += 1
        if 2 in extend_cover_index_list :
            all_ball_location = get_other_keys(now_area_loc[2], b_cover_num)
            search_list[2] = get_keys_from_dict(Ball_Dict_Load, all_ball_location, list(Ball_Dict_Load.keys()))
            b_cover_num += 1
        if 3 in extend_cover_index_list :
            all_opponent_location = get_other_keys(now_area_loc[3], c_cover_num)
            search_list[3] = get_keys_from_dict(Oppo_Dict_Load, all_opponent_location, list(Oppo_Dict_Load.keys()))
            c_cover_num += 1
        
        if t_cover_num > 2 or a_cover_num > 2 or b_cover_num > 2 or c_cover_num >  2:
            # print("Can not find index !!")
            t_cover_num = 1
            a_cover_num = 1
            b_cover_num = 1
            c_cover_num = 1
            break
            
    intersection = list(set.intersection(*map(set, search_list)))
    
    if len(intersection) < EXP_NUM:
        intersection.extend(random.sample(list(range(len(experience_trajectories))), EXP_NUM - len(intersection)))
    
    return intersection

def get_other_keys(location, cover_num):

    if location < 10:
        center_x = 0
        center_y = int(list(str(location))[0])
    else:
        center_x = int(list(str(location))[0])
        center_y = int(list(str(location))[1])

    if  cover_num == 1:
        if center_x == 9:
            right_x = center_x
        else:
            right_x = center_x + cover_num
        
        if center_x == 0:
            left_x = center_x
        else:
            left_x = center_x - cover_num
        
        if center_y == 9:
            right_y = center_y
        else:
            right_y = center_y + cover_num
        
        if center_y == 0:
            left_y = center_y
        else:
            left_y = center_y - cover_num

        
        keys_list = [str(left_x)+str(left_y), str(left_x)+str(center_y), str(left_x)+str(right_y), str(center_x)+str(left_y), str(center_x)+str(center_y), str(center_x)+str(right_y), str(right_x)+str(left_y), str(right_x)+str(center_y), str(right_x)+str(right_y)]
        keys_list = list(map(int, keys_list))
    
    if  cover_num == 2:
        if center_x == 9:
            right_x1 = center_x
            right_x2 = center_x
        elif center_x == 8:
            right_x1 = center_x + cover_num - 1
            right_x2 = center_x + cover_num - 1
        else:
            right_x1 = center_x + cover_num - 1
            right_x2 = center_x + cover_num
        
        if center_x == 0:
            left_x1 = center_x
            left_x2 = center_x
        elif center_x == 1:
            left_x1 = center_x - cover_num + 1
            left_x2 = center_x - cover_num + 1
        else:
            left_x1 = center_x - cover_num + 1
            left_x2 = center_x - cover_num
        
                
        if center_y == 9:
            right_y1 = center_y
            right_y2 = center_y
        elif center_y == 8:
            right_y1 = center_y + cover_num - 1
            right_y2 = center_y + cover_num - 1
        else:
            right_y1 = center_y + cover_num - 1
            right_y2 = center_y + cover_num
        
        if center_y == 0:
            left_y1 = center_y
            left_y2 = center_y
        elif center_y == 1:
            left_y1 = center_y - cover_num + 1
            left_y2 = center_y - cover_num + 1
        else:
            left_y1 = center_y - cover_num + 1
            left_y2 = center_y - cover_num

        keys_list0 = [str(center_x)+str(center_y)]
        keys_list1 = [str(left_x1)+str(left_y1), str(left_x1)+str(center_y), str(left_x1)+str(right_y1), str(center_x)+str(left_y1), str(center_x)+str(right_y1), str(right_x1)+str(left_y1), str(right_x1)+str(center_y), str(right_x1)+str(right_y1)]
        keys_list2 = [str(left_x2)+str(left_y2), str(left_x1)+str(left_y2), str(center_x)+str(left_y2), str(right_x1)+str(left_y2), str(right_x2)+str(left_y2)]
        keys_list3 = [str(left_x2)+str(left_y1), str(right_x2)+str(left_y1), str(left_x2)+str(center_y), str(right_x2)+str(center_y), str(left_x2)+str(right_y1), str(right_x2)+str(right_y1)]
        keys_list4 = [str(left_x2)+str(right_y2), str(left_x1)+str(right_y2), str(center_x)+str(right_y2), str(right_x1)+str(right_y2), str(right_x2)+str(right_y2)]
        keys_list = keys_list0 + keys_list1 + keys_list2 + keys_list3 + keys_list4
        keys_list = list(map(int, keys_list))
    
    return keys_list


def coord2area4gen(area_type, point):
    x = point[0]
    y = point[1]
    
    if type(x) != float and type(y) != float:
        try:
            x = x.numpy()
            y = y.numpy()
        except:
            x = x[0]
            y = y[0]

    area_loc = [-1,-1]
    if area_type == "player_location":
        cell_list_x = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        cell_list_y = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        area_loc[0] = int(bisect_left(cell_list_x, x))
        area_loc[1] = int(bisect_left(cell_list_y, y))
        
    if area_type == "ball_location": 
        cell_list_x = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        cell_list_y = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        area_loc[0] = int(bisect_left(cell_list_x, x))
        area_loc[1] = int(bisect_left(cell_list_y, y))
    
    if area_type == "opponent_location": 
        cell_list_x = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        cell_list_y = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        area_loc[0] = int(bisect_left(cell_list_x, x))
        area_loc[1] = int(bisect_left(cell_list_y, y))
    
    if area_loc[0] == 10:
        area_loc[0] = 5
    
    if area_loc[1] == 10:
        area_loc[1] = 5
    
    x_code = str(area_loc[0])
    y_code = str(area_loc[1])
    l_code = int(x_code + y_code)
    return l_code

def mapping_coord2area(player_current_state):
    now_catch_ball_type = player_current_state[12]
    now_player_location = coord2area4gen("player_location", [player_current_state[5], player_current_state[6]])
    now_ball_location = coord2area4gen("ball_location", [player_current_state[10], player_current_state[11]])
    now_opponent_location = coord2area4gen("opponent_location", [player_current_state[13], player_current_state[14]])
    now_ball_round = player_current_state[0]

    return [now_catch_ball_type, now_player_location, now_ball_location, now_opponent_location, now_ball_round]

def get_next_player_state(player_action, current_states, target_num):
    ret = torch.zeros((1, STATE_LEN))
    next_state = state_transformer(player_action.view(-1, ACTION_LEN).tolist()[0], current_states.view(-1, STATE_LEN).tolist()[0], target_num)
    ret[0,:] = torch.FloatTensor(next_state)
    return ret

def insert_sim_action_trajs(pass_traj_i):
    return experience_trajectories[pass_traj_i].copy()

def delete_state(ball):
    action_start_index = int((-1)*ACTION_LEN)
    return ball[action_start_index:]

def extract_sim_action_trajs(traj):
    traj = list(map(delete_state, traj))
    traj = torch.FloatTensor(traj)
    return traj

def prepare_act_input(pass_traj_list):
    if len(pass_traj_list) > EXP_NUM:
        pass_traj_list = random.sample(pass_traj_list, EXP_NUM)
    sim_action_trajs_list = list(map(insert_sim_action_trajs, pass_traj_list))
    sim_action_trajs_list = list(map(extract_sim_action_trajs, sim_action_trajs_list))
    return sim_action_trajs_list

def padding_function(traj):
    padding_length = 20 - traj.size(0)
    traj= F.pad(input=traj, pad=(0, 0, 0, padding_length), mode='constant', value=0)
    return traj

def prepare_enc_input(trajs_list_one_batch):
    trajs_list_one_batch = list(map(padding_function, trajs_list_one_batch))
    trajs_list_one_batch = torch.stack(trajs_list_one_batch)
    return trajs_list_one_batch