
import pickle
import argparse
import numpy as np
import pandas as pd
import sys
from bisect import bisect_left
from operator import itemgetter
from tqdm import tqdm, trange
from pickle import TRUE

parser = argparse.ArgumentParser()
parser.add_argument("--experience_from", type=str, help="path of the player's training rallies")
args = parser.parse_args()

match_state = ['rally', 'ball_round', 'player_score', 'opponent_score', 'score_status']
player_state = ['player_location_x', 'player_location_y']
ball_state = ['ball_distance','ball_distance_x','ball_distance_y',  'hit_x', 'hit_y']
opponent_state = ['opponent_type', 'opponent_location_x', 'opponent_location_y', 'opponent_move_x', 'opponent_move_y']
state_col = match_state + player_state + ball_state + opponent_state
action_col = ['landing_x', 'landing_y', 'player_type','player_move_location_x','player_move_location_y']
state_action_dividing = len(state_col)


def checkKey(our_dictionary, searching_key):
    if searching_key in our_dictionary.keys():
        return True
    else:
        return False

def checkVal(our_dictionary, searching_key, searching_value):
    if searching_value in our_dictionary[searching_key]:
        return True
    else:
        return False

def get_keys_from_dict(dictionary, keys, have_keys):
    print("have_keys ", have_keys)
    print("keys ", keys)
    intersection = list(set.intersection(*map(set, [keys, have_keys])))
    print("intersection ", intersection)
    out = itemgetter(*intersection)(dictionary)
    out = list(set.union(*map(set, out)))
    return out

def get_other_keys(location):
    
    center_x = int(list(str(location))[0])
    center_y = int(list(str(location))[1])
    
    if center_x == 9:
        rigth_x = center_x
    else:
        rigth_x = center_x + 1
    
    if center_x == 0:
        rigth_x = center_x
    else:
        left_x = center_x - 1

    
    if center_y == 9:
        rigth_y = center_y
    else:
        rigth_y = center_y + 1
    
    if center_y == 0:
        rigth_y = center_y
    else:
        left_y = center_y - 1

    keys_list = [str(left_x)+str(left_y), str(left_x)+str(center_y), str(left_x)+str(rigth_y), str(center_x)+str(left_y), str(center_x)+str(center_y), str(center_x)+str(rigth_y), str(rigth_x)+str(left_y), str(rigth_x)+str(center_y), str(rigth_x)+str(rigth_y)]
    keys_list = list(map(int, keys_list))
    return keys_list


def coord2area(area_type, point):
    x = point[0]
    y = point[1]
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
    x_code = str(area_loc[0])
    y_code = str(area_loc[1])
    l_code = int(x_code + y_code)
    return l_code


def main():
    
    experience_trajectories = pickle.load(open(args.experience_from, 'rb'))

    Shot_Dict = {}
    Play_Dict = {}
    Ball_Dict = {}
    Oppo_Dict = {}

    for rally_i in trange(len(experience_trajectories)):
        for ball_j in range(len(experience_trajectories[rally_i])):
            now_catch_ball_type = experience_trajectories[rally_i][ball_j][12]
            now_player_location = coord2area("player_location", experience_trajectories[rally_i][ball_j][5:7].copy())
            now_ball_location = coord2area("ball_location", experience_trajectories[rally_i][ball_j][10:12].copy())
            now_opponent_location = coord2area("opponent_location", experience_trajectories[rally_i][ball_j][13:15].copy())
            
            # To create a mapping dictionary, we will need to check whether the key and value already exist
            if checkKey(Shot_Dict, now_catch_ball_type):
                if checkVal(Shot_Dict, now_catch_ball_type, rally_i) == False:
                    Shot_Dict[now_catch_ball_type].append(rally_i)
            else:
                Shot_Dict[now_catch_ball_type] = [rally_i]

            if checkKey(Play_Dict, now_player_location):
                if checkVal(Play_Dict, now_player_location, rally_i) == False:
                    Play_Dict[now_player_location].append(rally_i)
            else:
                Play_Dict[now_player_location] = [rally_i]

            if checkKey(Ball_Dict, now_ball_location):
                if checkVal(Ball_Dict, now_ball_location, rally_i) == False:
                    Ball_Dict[now_ball_location].append(rally_i)
            else:
                Ball_Dict[now_ball_location] = [rally_i]

            if checkKey(Oppo_Dict, now_opponent_location):
                if checkVal(Oppo_Dict, now_opponent_location, rally_i) == False:
                    Oppo_Dict[now_opponent_location].append(rally_i)
            else:
                Oppo_Dict[now_opponent_location] = [rally_i]

    print("Shot_Dict")
    print(list(Shot_Dict.keys()))
    print("Play_Dict")
    print(list(Play_Dict.keys()))
    print("Ball_Dict")
    print(list(Ball_Dict.keys()))
    print("Oppo_Dict")
    print(list(Oppo_Dict.keys()))

    output = open('experience/Type_Dict.pkl', 'wb')
    pickle.dump(Shot_Dict, output)
    output.close()

    output = open('experience/Play_Dict.pkl', 'wb')
    pickle.dump(Play_Dict, output)
    output.close()

    output = open('experience/Ball_Dict.pkl', 'wb')
    pickle.dump(Ball_Dict, output)
    output.close()

    output = open('experience/Oppo_Dict.pkl', 'wb')
    pickle.dump(Oppo_Dict, output)
    output.close()


if __name__ == "__main__":
    main()