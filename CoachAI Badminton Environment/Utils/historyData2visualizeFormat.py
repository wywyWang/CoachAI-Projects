import pandas as pd
import numpy as np
import copy

type_mapping = {'發短球': 1, '長球': 2, '推撲球': 3, '殺球':4, '接殺防守':5, '平球':6, '網前球':7, '挑球':8, '切球':9, '發長球':10, '接不到':11} 

# can only deal with one set
df = pd.read_csv('ChouTeinChen.csv')

df = df[['player','rally_id','type','rally','ball_round', 'landing_region','player_region','opponent_region']]

output = copy.deepcopy(df)

output =  output.rename(columns={'player':'obs_player', 'type':'act_ball_type', 'ball_round':'obs_ball_round'})
#output['act_ball_type'] = output['act_ball_type'].replace(type_mapping)

output['obs_player'] = output['obs_player'].apply(lambda x: 'B' if x == 'CHOU Tien Chen' else 'A')

output['obs_serve'] = output['obs_ball_round'].apply(lambda x: True if x == 0 else False)

output['rally'] = output['rally_id']

output.to_csv('ChouTeinChen_act.csv',index=False)