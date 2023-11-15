import pandas as pd
import numpy as np
import json

# shot type (1,2,3,4,5,6,7,8,9,10,11)
# 11 -> cannot reach
def build_table():
    data = pd.read_csv('D:\文件\學校\專題\ShuttleNetVisualize\StrokeForecasting\data\dicrete_cannotReach.csv')
    #data1 = pd.read_csv('updated_data.csv')
    used_column = ['rally_id', 'type', 'landing_region', 'player_region', 'opponent_region']
    type_mapping = {'發短球': 0, '長球': 1, '推撲球': 2, '殺球':3, '接殺防守':4, '平球':5, '網前球':6, '挑球':7, '切球':8, '發長球':9 , '接不到':10} 
    data = data[used_column]

    data['shot'] = data['type']
    data['land'] = data['landing_region']
    data['shot'] = data['shot'].replace(type_mapping)

    grouped = data.groupby('rally_id')

    for name, group in grouped:
        for i in range(0, len(group)-1):
            data['move'] = grouped['opponent_region'].shift(-1)


    data.to_csv('label_action.csv', index=False)

if __name__ == "__main__":
    build_table()
    data = pd.read_csv('label_action.csv')
    data['move'] = data['move'].replace(np.nan, 5)
    data.to_csv('label_action.csv', index=False)