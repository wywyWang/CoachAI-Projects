#%%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from pickle import TRUE
import pickle
import numpy as np
import pandas as pd
import os
import random
import sys
import csv
import json
import seaborn as sns
from math import log2
from math import sqrt
from math import ceil
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import sys
import functools
import operator
#%%
match_state = ['rally', 'ball_round', 'player_score', 'opponent_score', 'score_status']
player_state = ['player_location_x', 'player_location_y']
ball_state = ['ball_distance','ball_distance_x','ball_distance_y',  'hit_x', 'hit_y']
opponent_state = ['opponent_type', 'opponent_location_x', 'opponent_location_y', 'opponent_move_x', 'opponent_move_y']
state_col = match_state + player_state + ball_state + opponent_state
player_col = ['player']
action_col = ['landing_x', 'landing_y', 'player_type','moving_x','moving_y']
state_action_dividing = len(state_col)

testing_trajectories = pickle.load(open('Data/0/player_test_0.pkl', 'rb'))
ans_list = []
target_players = ['PUSARLA V. Sindhu', 'NG Ka Long Angus', 'Pornpawee CHOCHUWONG', 'Neslihan YIGIT', 'Viktor AXELSEN', 'LIEW Daren', 'Jonatan CHRISTIE', 'LEE Zii Jia', 'Anders ANTONSEN', 'Carolina MARIN', 'Kento MOMOTA', 'CHEN Yufei', 'Hans-Kristian Solberg VITTINGHUS', 'WANG Tzu Wei', 'LEE Cheuk Yiu', 'An Se Young', 'Rasmus GEMKE', 'Evgeniya KOSETSKAYA', 'Sameer VERMA', 'Khosit PHETPRADAB', 'KIDAMBI Srikanth', 'TAI Tzu Ying', 'Supanida KATETHONG', 'Busanan ONGBAMRUNGPHAN', 'CHEN Long', 'Ratchanok INTANON', 'CHOU Tien Chen', 'Michelle LI', 'Anthony Sinisuka GINTING', 'Mia BLICHFELDT', 'SHI Yuqi']
statistic_name = 'CHOU Tien Chen'
for i in range(len(testing_trajectories)):
    if testing_trajectories[i][0][17] == target_players.index(statistic_name):
        ans_list.append(testing_trajectories[i][:-1].copy())
print(len(ans_list))

testing_trajectories = pickle.load(open('report_synthetic_player_match_result.pkl', 'rb'))
gen_list = []
target_players = ['PUSARLA V. Sindhu', 'NG Ka Long Angus', 'Pornpawee CHOCHUWONG', 'Neslihan YIGIT', 'Viktor AXELSEN', 'LIEW Daren', 'Jonatan CHRISTIE', 'LEE Zii Jia', 'Anders ANTONSEN', 'Carolina MARIN', 'Kento MOMOTA', 'CHEN Yufei', 'Hans-Kristian Solberg VITTINGHUS', 'WANG Tzu Wei', 'LEE Cheuk Yiu', 'An Se Young', 'Rasmus GEMKE', 'Evgeniya KOSETSKAYA', 'Sameer VERMA', 'Khosit PHETPRADAB', 'KIDAMBI Srikanth', 'TAI Tzu Ying', 'Supanida KATETHONG', 'Busanan ONGBAMRUNGPHAN', 'CHEN Long', 'Ratchanok INTANON', 'CHOU Tien Chen', 'Michelle LI', 'Anthony Sinisuka GINTING', 'Mia BLICHFELDT', 'SHI Yuqi']
statistic_name = 'CHOU Tien Chen' # TAI Tzu Ying CHOU Tien Chen
for i in range(len(testing_trajectories)):
    if testing_trajectories[i][0][17] == target_players.index(statistic_name):
        gen_list.append(testing_trajectories[i][:-1].copy())
print(len(gen_list))


"""
{'發短球': 1, '發長球': 2, '長球': 3, '殺球': 4, '切球': 5, '挑球': 6,
'平球': 7, '網前球': 8, '推撲球': 9, '接殺防守': 10}
"""

# 4
# gen_bw_adjust_vol = 0.6
# ans_bw_adjust_vol = 0.6
# gen_thresh = 0.3
# ans_thresh = 0.1
# ANS = True
# GEN = True
# DRAW = True
# ACT = "move"

# 3
# gen_bw_adjust_vol = 0.6
# ans_bw_adjust_vol = 0.6
# gen_thresh = 0.1
# ans_thresh = 0.4

# 2KDD
# gen_bw_adjust_vol = 1
# ans_bw_adjust_vol = 1
# gen_thresh = 0.5
# ans_thresh = 0.1

# 2NIPS
gen_bw_adjust_vol = 1.2
ans_bw_adjust_vol = 1
gen_thresh = 0.7
ans_thresh = 0

ANS = True
GEN = True
DRAW = True


# gen_info = random.sample(gen_list, int(len(ans_list)))
gen_info = gen_list.copy()
ans_info = ans_list.copy()

gen_info = functools.reduce(operator.concat,gen_info)
ans_info = functools.reduce(operator.concat,ans_info)
gen_df = pd.DataFrame( gen_info, columns = state_col + player_col + action_col )
ans_df = pd.DataFrame( ans_info, columns = state_col + player_col + action_col )

gen_df['landing_x'] = (gen_df['landing_x'] * (355/2))
gen_df['landing_y'] = (gen_df['landing_y'] * 240) + 120
gen_df['moving_x'] = (gen_df['moving_x'] * (355/2))
gen_df['moving_y'] = (gen_df['moving_y'] * 240) + 120

ans_df['landing_x'] = (ans_df['landing_x'] * (355/2))
ans_df['landing_y'] = (ans_df['landing_y'] * 240) + 120
ans_df['moving_x'] = (ans_df['moving_x'] * (355/2))
ans_df['moving_y'] = (ans_df['moving_y'] * 240) + 120


figsize = (14, 24)

def show_cor(data, answer, x, y, row, col, rowspan, colspan, xlim, ylim, hue=None):
    print(len(data))
    fig_x = plt.subplot2grid(figsize, (row, col), colspan=colspan-1, xlim=xlim, xticks=[], yticks=[], frame_on=False)
    fig_y = plt.subplot2grid(figsize, (row+1, col+colspan-1), rowspan=rowspan-1, ylim=ylim, xticks=[], yticks=[], frame_on=False)
    # fig = plt.subplot2grid(figsize, (row+1, col), rowspan=rowspan-1, colspan=colspan-1, xlim=xlim, ylim=ylim)
    fig = plt.subplot2grid(figsize, (row+1, col), rowspan=rowspan-1, colspan=colspan-1, xlim=xlim, ylim=ylim)
    try:
        # print(data)
        if DRAW == False:
            1/0
        else:
            if ANS == True:
                # ax=fig_x, ax=fig_y, ax=fig, 
                # sns.kdeplot(data=answer, x=x, hue=hue, legend=False, bw_adjust = ans_bw_adjust_vol, levels=10, thresh=ans_thresh)
                # sns.kdeplot(data=answer, y=y, hue=hue, legend=False, bw_adjust = ans_bw_adjust_vol, levels=10, thresh=ans_thresh)
                sns.kdeplot(data=answer, x=x, y=y, hue=hue, legend=False, bw_adjust = ans_bw_adjust_vol, levels=10, thresh=ans_thresh, label="correct distribution")
            if GEN == True: # ax=fig_x, ax=fig_y, ax=fig, 
                # sns.kdeplot(data=data, x=x, hue=hue, legend=False, bw_adjust = gen_bw_adjust_vol, levels=10, thresh=gen_thresh)
                # sns.kdeplot(data=data, y=y, hue=hue, legend=False, bw_adjust = gen_bw_adjust_vol, levels=10, thresh=gen_thresh)
                sns.kdeplot(data=data, x=x, y=y, hue=hue, legend=False, bw_adjust = gen_bw_adjust_vol, levels=10, thresh=gen_thresh, label="generated distribution")
    except:
        if ANS == True:
            sns.scatterplot(ax=fig, data=answer, x=x, y=y, hue=hue)
        else:
            sns.scatterplot(ax=fig, data=data, x=x, y=y, hue=hue)
    
    x = 177.5
    y = 480

    fig.plot([25-x, 330-x], [810-y, 810-y], color='black', linestyle='-', linewidth=1)
    fig.plot([25-x, 330-x], [756-y, 756-y], color='black', linestyle='-', linewidth=1)
    fig.plot([25-x, 330-x], [594-y, 594-y], color='black', linestyle='-', linewidth=1)
    fig.plot([25-x, 330-x], [366-y, 366-y], color='black', linestyle='-', linewidth=1)
    fig.plot([25-x, 330-x], [204-y, 204-y], color='black', linestyle='-', linewidth=1)
    fig.plot([25-x, 330-x], [150-y, 150-y], color='black', linestyle='-', linewidth=1)
    fig.plot([25-x, 25-x],  [150-y, 810-y], color='black', linestyle='-', linewidth=1)
    fig.plot([50-x, 50-x],  [150-y, 810-y], color='black', linestyle='-', linewidth=1)
    fig.plot([177.5-x, 177.5-x], [150-y, 810-y], color='black', linestyle='-', linewidth=1)
    fig.plot([305-x, 305-x], [150-y, 810-y], color='black', linestyle='-', linewidth=1)
    fig.plot([330-x, 330-x], [150-y, 810-y], color='black', linestyle='-', linewidth=1)
    fig.plot([25-x, 330-x],  [480-y, 480-y], color='black', linestyle='-', linewidth=1) 


t = 8
p = 6

ACT = "land"
gen = gen_df
ans = ans_df

gen = gen_df[gen_df['player_type'] == t]
ans = ans_df[ans_df['player_type'] == t]
# gen = gen[gen['opponent_type'] == p]
# ans = ans[ans['opponent_type'] == p]

# gen = gen[gen['player_location_y'] > 0]
# ans = ans[ans['player_location_y'] > 0]
# gen = gen[gen['player_location_x'] > 0]
# ans = ans[ans['player_location_x'] > 0]
# gen = gen[gen['opponent_location_y'] > 0]
# ans = ans[ans['opponent_location_y'] > 0]
# gen = gen[gen['opponent_location_x'] < 0]
# ans = ans[ans['opponent_location_x'] < 0]


gen = gen.loc[~((gen['landing_x'] == 0) | (gen['landing_y'] == 0))]
gen = gen.loc[~((gen['moving_x'] == 0) | (gen['moving_y'] == 0))]

print(len(gen), len(ans))

print("start draw landing area")
if ACT == "land":
    show_cor(gen, ans, 'landing_x', 'landing_y', 0, 0, 15, 15, (-177.5, 177.5), (0, 480))
else:
    show_cor(gen, ans, 'moving_x', 'moving_y', 0, 0, 15, 15, (-177.5, 177.5), (0, 480))

plt.legend(loc='upper right')
# plt.legend(labels=["generated distribution","correct distribution"])
plt.show()

if ACT == "move":
    plt.savefig("moving.png")
else:
    plt.savefig("landing.png")
# %%
