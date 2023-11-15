import numpy as np
import pandas as pd
from typing import Literal, Tuple, Dict

def obj2subj_coord(objective_coord: Tuple[float, float], player: Literal[1, 2]):
    x, y = objective_coord
    y = 960 - y  # move origin to left bottom
    if player == 2:
        x -= 177.5
        y -= 480
    elif player == 1:
        # rotate 180 deg
        x = 355 - x
        y = 960 - y
        x -= 177.5
        y -= 480
    else:
        NotImplementedError
    return x, y
    

data = pd.read_csv('2023dataset\data\datasetRemoveBad.csv')
#used_column = ['rally_id','type','player','ball_round','set','match_id','rally_id','landing_x', 'landing_y', 'player_location_x', 'player_location_y', 'opponent_location_x', 'opponent_location_y','type']
#data = data[used_column]

# remove empty data
for rally_id in data[data['opponent_location_x'].isna()]['rally_id']:
    drop_count = len(data[data['rally_id'] == rally_id])
    print(f'drop {drop_count} rows')
    data = data[data['rally_id'] != rally_id]

mean_x, std_x = 175., 82.
mean_y, std_y = 467., 192.
data['landing_x'] = (data['landing_x']-mean_x) / std_x
data['landing_y'] = (data['landing_y']-mean_y) / std_y
data['player_location_x'] = (data['player_location_x']-mean_x) / std_x
data['player_location_y'] = (data['player_location_y']-mean_y) / std_y
data['opponent_location_x'] = (data['opponent_location_x']-mean_x) / std_x
data['opponent_location_y'] = (data['opponent_location_y']-mean_y) / std_y

new_rows: Dict[str, list] = {'landing_x':[], 'landing_y': [], 'player_location_x':[], 'player_location_y':[], 
                             'opponent_location_x':[], 'opponent_location_y':[],'type':[], 'rally' :[],
                             'rally_id' : [], 'ball_round': [],'player':[], 'set':[], 'match_id':[]}
for i, row in data.iterrows():
    if row['win_reason'] == '落地致勝':
        new_rows['landing_x'].append(0.)
        new_rows['landing_y'].append(0.)
        new_rows['player_location_x'].append(0.)
        new_rows['player_location_y'].append(1. if row['player_location_y'] < 0 else -1.)
        new_rows['opponent_location_x'].append(0.)
        new_rows['opponent_location_y'].append(1. if row['opponent_location_y'] < 0 else -1.)
        new_rows['type'].append('接不到')
        new_rows['rally'].append(row['rally'])
        new_rows['rally_id'].append(row['rally_id'])
        new_rows['ball_round'].append(row['ball_round']+1)
        new_rows['player'].append(data.at[i-1, 'player'])
        new_rows['set'].append(row['set'])
        new_rows['match_id'].append(row['match_id'])


data = pd.concat([data, pd.DataFrame(new_rows)])#.reset_index(drop=True)

data = data.sort_values(by=['rally_id','rally', 'ball_round'])

data = data.drop_duplicates()

data.to_csv('continous_cannotReach_trainingData.csv', index=False)

