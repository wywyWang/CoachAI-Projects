import numpy as np
import pandas as pd

def continue2discrete_coordX(x):
    if x < 50:    return 0. #outside
    elif x < 135: return 92.5
    elif x < 215: return 115.
    elif x < 305: return 260.
    else:         return 310.  #outside

def continue2discrete_coordY(y):
    # y in [0, 960] [150,810]
    if y < 150:   return 100.
    elif y < 260: return 205.
    elif y < 370: return 315.
    elif y < 480: return 425.
    elif y < 590: return 535.
    elif y < 700: return 645.
    elif y < 810: return 755.
    else:         return 900.
    

data = pd.read_csv('2023dataset\data\datasetRemoveBad.csv')

data['landing_x'] = data['landing_x'].apply(lambda x : continue2discrete_coordX(x))
data['landing_y'] = data['landing_y'].apply(lambda y : continue2discrete_coordY(y))
data['player_location_x'] = data['player_location_x'].apply(lambda x : continue2discrete_coordX(x))
data['player_location_y'] = data['player_location_y'].apply(lambda y : continue2discrete_coordY(y))
data['opponent_location_x'] = data['opponent_location_x'].apply(lambda x : continue2discrete_coordX(x))
data['opponent_location_y'] = data['opponent_location_y'].apply(lambda y : continue2discrete_coordY(y))
data['hit_x'] = data['hit_x'].apply(lambda x : continue2discrete_coordX(x))
data['hit_y'] = data['hit_y'].apply(lambda y : continue2discrete_coordY(y))
mean_x, std_x = 175., 82.
mean_y, std_y = 467., 192.
data['landing_x'] = (data['landing_x']-mean_x) / std_x
data['landing_y'] = (data['landing_y']-mean_y) / std_y

data.to_csv('discrete_shuttleNetInput.csv', index=False)



