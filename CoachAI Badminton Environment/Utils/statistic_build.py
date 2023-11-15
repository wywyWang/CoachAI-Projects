import numpy as np
import pandas as pd

"""
Convert (x,y) coord to region (from 1 to 10)
"""
def getRegion(x, y):
    # x in [50 305] -> [50 135 215 305]
    # y in [150 480] -> [150 260 370 480]
    if y < 150:   return 10
    elif y < 260: region = [7, 8, 9]
    elif y < 370: region = [4, 5, 6]
    elif y < 480: region = [1, 2, 3]
    else:         return 10


    if x < 50:    return 10 #outside
    elif x < 135: return region[0]
    elif x < 215: return region[1]
    elif x < 305: return region[2]
    else:         return 10  #outside
    

data = pd.read_csv('2023dataset\data\datasetRemoveBad.csv')
#used_column = ['rally_id','type','player','ball_round','set','match_id','rally_id','landing_x', 'landing_y', 'player_location_x', 'player_location_y', 'opponent_location_x', 'opponent_location_y','type']
#data = data[used_column]

new_rows = {'landing_region':[], 'player_region':[], 'opponent_region':[], 'type':[], 'rally' :[],
            'rally_id' : [], 'ball_round': [],'player':[], 'set':[]}
for i, row in data.iterrows():
    if row['player_location_y'] < (960/2): # net
        row['landing_y'] = 960 - row['landing_y']
        row['landing_x'] = 355 - row['landing_x']
        row['opponent_location_y'] = 960 - row['opponent_location_y']
        row['opponent_location_x'] = 355 - row['opponent_location_x']
    else:
        row['player_location_y'] = 960 - row['player_location_y']
        row['player_location_x'] = 355 - row['player_location_x']
    #data.at[i, 'landing_y'] = row['landing_y']
    #data.at[i, 'landing_x'] = row['landing_x']
    #data.at[i, 'opponent_location_y'] = row['opponent_location_y']
    #data.at[i, 'opponent_location_x'] = row['opponent_location_x']
    #data.at[i, 'player_location_y'] = row['player_location_y']
    #data.at[i, 'player_location_x'] = row['player_location_x']

    data.at[i, 'landing_region'] = getRegion(row['landing_x'], row['landing_y'])
    data.at[i, 'player_region'] = getRegion(row['player_location_x'], row['player_location_y'])
    data.at[i, 'opponent_region'] = getRegion(row['opponent_location_x'], row['opponent_location_y'])

    if(row['win_reason'] == '落地致勝'):
        new_rows['landing_region'].append(5)
        new_rows['player_region'].append(5)
        new_rows['opponent_region'].append(5)
        new_rows['type'].append('接不到')
        new_rows['rally'].append(row['rally'])
        new_rows['rally_id'].append(row['rally_id'])
        new_rows['ball_round'].append(row['ball_round']+1)
        new_rows['player'].append(data.at[i-1, 'player'])
        new_rows['set'].append(row['set'])
        new_row = pd.DataFrame({'landing_region':5, 'player_region':5, 'opponent_region':5, 'type':'接不到', 'rally' : row['rally'],
                        'rally_id' : row['rally_id'], 'ball_round': row['ball_round']+1,'player':data.at[i-1, 'player'], 'set':row['set']}, index = [i])


data = pd.concat([data, pd.DataFrame(new_rows)])#.reset_index(drop=True)

data = data.sort_values(by=['rally_id','rally', 'ball_round'])
    
data[['landing_region', 'player_region', 'opponent_region']] = data[['landing_region', 'player_region', 'opponent_region']].astype(int)

data.to_csv('dicrete_cannotReach.csv', index=False)



