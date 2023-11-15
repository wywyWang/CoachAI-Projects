import pandas as pd
import numpy as np


df = pd.read_csv('ChouTeinChen.csv')

df = df[['lose_reason', 'win_reason', 'getpoint_player','player']]
df = df[df['lose_reason'].notnull()]

win_reason = {'ball land':0, 'return fail':0, 'outside':0}
for index, row in df.iterrows():
    if row['player'] != 'CHOU Tien Chen':
        if row['win_reason'] == '對手落點判斷失誤':
            win_reason['ball land'] += 1
        elif row['win_reason'] == '落地致勝':
            win_reason['ball land'] += 1
        elif row['win_reason'] == '對手掛網':
            win_reason['return fail'] += 1
        elif row['win_reason'] == '對手未過網':
            win_reason['ball land'] += 1
        elif row['win_reason'] == '對手出界':
            win_reason['outside'] += 1
        elif row['win_reason'] == '落地判斷失誤':
            win_reason['ball land'] += 1
        else:
            print( row['win_reason'])
        

print(win_reason)
df.to_csv('ChouTeinChen_output.csv')