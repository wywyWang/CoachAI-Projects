import pandas as pd
import numpy as np
import json

# shot type (1,2,3,4,5,6,7,8,9,10,11)
# 11 -> cannot reach
def build_table():
    data = pd.read_csv('MDP.csv')
    used_column = ['rally_id', 'type', 'landing_region', 'player_region', 'opponent_region']
    type_mapping = {'發短球': 0, '長球': 1, '推撲球': 2, '殺球':3, '接殺防守':4, '平球':5, '網前球':6, '挑球':7, '切球':8, '發長球':9} 
    data = data[used_column]
    data['type'] = data['type'].replace(type_mapping)
    data['type'] = data['type'].astype(int)

    grouped = data.groupby('rally_id')

    table = np.zeros([10, 10, 10, 11, 10, 10], dtype=np.float32) #((player, opponent, ball), (shot, land, move))
    print(table.shape)

    for name, group in grouped:
        for i in range(1, len(group)-1): # start from 1 to avoid service ball
            current_player = group.iloc[i]['player_region']-1
            current_opponent = group.iloc[i]['opponent_region']-1
            current_landing = group.iloc[i]['landing_region']-1
            current_type = group.iloc[i]['type'] # -1 to remove launch ball

            next_player = group.iloc[i+1]['player_region']-1
            next_opponent = group.iloc[i+1]['opponent_region']-1
            next_landing = group.iloc[i+1]['landing_region']-1

            # state( player, opponent, ball)
            if i == 0:
                state = (current_player, current_opponent, current_player)
            else:
                last_landing = group.iloc[i]['landing_region']-1
                state = (current_player, current_opponent, last_landing)

            action = (current_type, current_landing, next_opponent)

            # table[current_player-1][current_landing-1][current_landing-1][next_player-1] += 1
            table[state][action] += 1
            # print(current_player, current_opponenet, current_landing)
        
    table /= table.sum(axis=(-3,-2,-1), keepdims=True)
    np.save("MDP_table.npy", table)
    tablelist = table.tolist()
    with open('table.json', 'w') as f:
        json.dump(tablelist, f)

if __name__ == "__main__":
    build_table()