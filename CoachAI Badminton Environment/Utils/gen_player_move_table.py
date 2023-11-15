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

    grouped = data.groupby('rally_id')

    table = np.zeros([11,10,10,10], dtype=np.float32) #(player, opponent, ball, next_player)
    print(table.shape)

    for name, group in grouped:
        for i in range(0, len(group)-1):
            current_player = group.iloc[i]['player_region']
            current_opponent = group.iloc[i]['opponent_region']
            current_landing = group.iloc[i]['landing_region']

            next_player = group.iloc[i+1]['player_region']

            table[current_player-1][current_landing-1][current_landing-1][next_player-1] += 1
            # print(current_player, current_opponenet, current_landing)
        
    table /= table.sum(axis=3, keepdims=True)
    np.save("next_player_move_table.npy", table)
    #tablelist = table.tolist()
    #with open('table.json', 'w') as f:
    #    json.dump(tablelist, f)

if __name__ == "__main__":
    build_table()