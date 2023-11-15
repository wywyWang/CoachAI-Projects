import pandas as pd
import numpy as np

data = pd.read_csv('StrokeForecasting/data/dicrete_cannotReach.csv')
data = data[['rally_id','type','rally','ball_round', 'landing_region','player_region','opponent_region']]
grouped = data.groupby(['rally_id'])
filtered = grouped.filter(lambda x: len(x) >= 5)
result = filtered.groupby(['rally_id']).head(5)
group_keys = list(result.groupby(['rally_id']).groups.keys())

type_mapping = type_mapping = {'發短球': 1, '長球': 2, '推撲球': 3, '殺球':4, '接殺防守':5, '平球':6, '網前球':7, '挑球':8, '切球':9, '發長球':10, '接不到':11} 

"""
    state: (player, opponent, ball)
    action: (shot, land, move)
"""

def sample():
    random_group_index = np.random.choice(len(group_keys))
    rows = result.groupby(['rally_id']).get_group(group_keys[random_group_index])
    states = []
    actions = []

    for i, (index, row) in enumerate(rows.iterrows()):
        player_region = row['player_region']
        opponent_region = row['opponent_region']
        landing_region = row['landing_region']
        type = type_mapping[row['type']]

        if i == 0:
            state = (player_region, opponent_region, player_region)
            states.append(state)
        else:
            state = (player_region, opponent_region, prev_landing_region)
            action = (prev_type, prev_landing_region, opponent_region)
            states.append(state)
            actions.append(action)


        prev_landing_region = landing_region
        prev_opponent_region = opponent_region
        prev_type = type

    print(rows)
    print(states)
    print(actions)

sample()