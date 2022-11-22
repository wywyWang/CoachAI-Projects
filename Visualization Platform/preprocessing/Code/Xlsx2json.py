import pandas as pd
import json
import numpy as np
from collections import Counter
import itertools  as it

#Process rally type
#### May need to modify file path and file name ####
filename = './clip_info_19ASI_CS'
filepath = './Statistics/rally_type_real_' + filename.split('_')[-1] + '.json'

rally = pd.read_excel(filename + '.xlsx')
rally = rally[['unique_id','rally','ball_round','player','frame_num','server','type','lose_reason','roundscore_A','roundscore_B']]
rally_drop = rally.drop(["frame_num","server","lose_reason"],axis=1)
totaltype = ["挑球","防守回挑","放小球","擋小球","發小球","殺球","切球","過渡切球","長球","發長球","平球","撲球"]
unneedtype = ['未擊球','未過網','掛網球']

for i in range(len(rally_drop['unique_id'])):
    rally_drop['unique_id'][i] = rally_drop['unique_id'][i].split('-')[-1]
    
field = rally_drop.groupby(['unique_id','rally'])
result = pd.DataFrame(columns = ["set","rally","player","balltype","count"])

roundset = 1
roundcnt = 1

for i in list(field.groups.keys()):
    score = field.get_group(i)[['roundscore_A','roundscore_B']].iloc[-1]
    if int(score[0]) >=21 or int(score[1]) >=21:
        if abs(int(score[0]) - int(score[1])) == 2:
            roundset +=1
            roundcnt = 1
    
    resultA = pd.DataFrame(columns = ["set","rally","player","balltype","count"])
    resultB = pd.DataFrame(columns = ["set","rally","player","balltype","count"])
    
    print("========================")
    print("set = ",roundset)
    print("cnt = ",roundcnt)
    tmp = pd.Series.to_frame(field.get_group(i).groupby('player')['type'].value_counts())

    tmptolist = tmp.index.tolist()
    ballset = []
    ballround = []
    player = []
    balltype = []
    ballsum = []
    for j in range(len(tmptolist)):
        if tmptolist[j][1] not in unneedtype:
            ballset += [str(roundset)]
            ballround += [str(roundcnt)]
            player += [tmptolist[j][0]]
            balltype += [tmptolist[j][1]]
            ballsum += [tmp['type'][j]]

    print("player A = ",Counter(player)['A'])
    print("player B = ",Counter(player)['B'])
    bsetA = ballset[:Counter(player)['A']]
    brA = ballround[:Counter(player)['A']]
    plA = player[:Counter(player)['A']]
    btA = balltype[:Counter(player)['A']]
    bsA = ballsum[:Counter(player)['A']]
    # 放A因為前半段是A 後半段是B
    bsetB = ballset[Counter(player)['A']:]
    brB = ballround[Counter(player)['A']:]
    plB = player[Counter(player)['A']:]
    btB = balltype[Counter(player)['A']:]
    bsB = ballsum[Counter(player)['A']:]

    for item in totaltype:
        if item not in btA:
            bsetA += [str(roundset)]
            brA += [str(roundcnt)]
            plA += ['A']
            btA += [item]
            bsA += [0]
        if item not in btB:
            bsetB += [str(roundset)]
            brB += [str(roundcnt)]
            plB += ['B']
            btB += [item]
            bsB += [0]
            
    roundcnt += 1

    resultA['set'] = bsetA
    resultA['rally'] = brA
    resultA['player'] = plA
    resultA['balltype'] = btA
    resultA['count'] = bsA
    resultA = resultA.sort_values(['balltype'])
    result = result.append(resultA)

    resultB['set'] = bsetB
    resultB['rally'] = brB
    resultB['player'] = plB
    resultB['balltype'] = btB
    resultB['count'] = bsB
    resultB = resultB.sort_values(['balltype'])
    result = result.append(resultB)

result = (result.groupby(['set','rally','player'], as_index=False)
            .apply(lambda x: x[['balltype','count']].to_dict('r'))
            .reset_index()
            .rename(columns={0:'result'})
            )
result = (result.groupby(['set'], as_index=False)
            .apply(lambda x: x[['rally','player','result']].to_dict('r'))
            .reset_index()
            .rename(columns={0:'info'})
            )

export_json(filepath,result)

#Process our predict data
rally = pd.read_excel('./clip_info_new.xlsx')
rally = rally[['unique_id','rally','ball_round','player','frame_num','server','prediction','lose_reason']]

conv_balltype = { 
    'cut': '切球', 
    'drive': '平球', 
    'lob': '挑球' , 
    'long': '長球', 
    'netplay': '小球',
    'rush': '撲球',
    'smash': '殺球'
}
rally['prediction'] = rally['prediction'].map(conv_balltype)

rally_drop = rally.drop(["frame_num","server","lose_reason"],axis=1)
rally_drop = rally_drop.dropna(subset=['prediction']).reset_index(drop=True)
totaltype = ["挑球","切球","長球","殺球","平球","撲球","小球"]

for i in range(len(rally['unique_id'])):
    rally['unique_id'][i] = rally['unique_id'][i].split('-')[-2] + '-' + rally['unique_id'][i].split('-')[-1]

field = rally_drop.groupby(['unique_id','rally'])
filepath = './rally_type.json'

print(rally['prediction'])
result = pd.DataFrame(columns = ["set","rally","player","balltype","count"])

roundset = 1
roundcnt = 1
for i in list(field.groups.keys()):
#     if roundcnt > 25:
#         roundset = 2
#         roundcnt = 1
    resultA = pd.DataFrame(columns = ["set","rally","player","balltype","count"])
    resultB = pd.DataFrame(columns = ["set","rally","player","balltype","count"])
    print("========================")
    tmp = pd.Series.to_frame(field.get_group(i).groupby('player')['prediction'].value_counts())

    tmptolist = tmp.index.tolist()
    ballset = []
    ballround = []
    player = []
    balltype = []
    ballsum = []
    for j in range(len(tmptolist)):
        ballset += [str(roundset)]
        ballround += [str(roundcnt)]
        player += [tmptolist[j][0]]
        balltype += [tmptolist[j][1]]
        ballsum += [tmp['prediction'][j]]

    bsetA = ballset[:Counter(player)['A']]
    brA = ballround[:Counter(player)['A']]
    plA = player[:Counter(player)['A']]
    btA = balltype[:Counter(player)['A']]
    bsA = ballsum[:Counter(player)['A']]
    # 放A因為前半段是A 後半段是B
    bsetB = ballset[Counter(player)['A']:]
    brB = ballround[Counter(player)['A']:]
    plB = player[Counter(player)['A']:]
    btB = balltype[Counter(player)['A']:]
    bsB = ballsum[Counter(player)['A']:]


    for item in totaltype:
        if item not in btA:
            bsetA += [str(roundset)]
            brA += [str(roundcnt)]
            plA += ['A']
            btA += [item]
            bsA += [0]
        if item not in btB:
            bsetB += [str(roundset)]
            brB += [str(roundcnt)]
            plB += ['B']
            btB += [item]
            bsB += [0]

    roundcnt += 1

    resultA['set'] = bsetA
    resultA['rally'] = brA
    resultA['player'] = plA
    resultA['balltype'] = btA
    resultA['count'] = bsA
    resultA = resultA.sort_values(['balltype'])
    result = result.append(resultA)

    resultB['set'] = bsetB
    resultB['rally'] = brB
    resultB['player'] = plB
    resultB['balltype'] = btB
    resultB['count'] = bsB
    resultB = resultB.sort_values(['balltype'])

    result = result.append(resultB)

result = (result.groupby(['set','rally','player'], as_index=False)
            .apply(lambda x: x[['balltype','count']].to_dict('r'))
            .reset_index()
            .rename(columns={0:'result'})
            )
result = (result.groupby(['set'], as_index=False)
            .apply(lambda x: x[['rally','player','result']].to_dict('r'))
            .reset_index()
            .rename(columns={0:'info'})
            )

export_json(filepath,result)

#Process rally count real
#### May need to modify file path and file name ####
filename = './clip_info_18IND_TC'
filepath = './Statistics/rally_count_real_' + filename.split('_')[-1] + '.json'

rally = pd.read_excel(filename + '.xlsx')
rally = rally[['unique_id','rally','ball_round','getpoint_player','lose_reason','roundscore_A','roundscore_B']]

for i in range(len(rally['unique_id'])):
    rally['unique_id'][i] = rally['unique_id'][i].split('-')[-1]    

field = rally.groupby(['unique_id','rally'])

set = []
rallys = []
hit_number = []
score = []

winner = rally['getpoint_player']
winner = winner.dropna().reset_index(drop=True)
result = pd.DataFrame(columns = ["set","rally","score","stroke","winner","on_off_court","balltype","lose_area"])

score_idx = -1
for i in range(len(field.groups.keys())):
    setrally = list(field.groups.keys())[i][0]
    set += [int(setrally)]
    rallys +=[list(field.groups.keys())[i][1]]
    hit_count = len(field.get_group(list(field.groups.keys())[i])['ball_round'].value_counts())-1
    hit_number += [hit_count]
    score_idx += (hit_count+1)
    score_detail = str(rally['roundscore_A'][score_idx]) + ':' + str(rally['roundscore_B'][score_idx])
    score += [score_detail]
    
result['set'] = set
result['rally'] = rallys
result['stroke'] = hit_number
result['winner'] = winner
result['score'] = score

rally2 = pd.read_excel(filename + '.xlsx')
rally2 = rally2[['hit_area','lose_reason']].dropna().reset_index(drop=True)
result['on_off_court'] = rally2['lose_reason']
result['lose_area'] = rally2['hit_area']

rally2 = pd.read_excel(filename + '.xlsx')
rally2 = rally2[['type','getpoint_player']]
balltype = []
for i in range(len(rally2)):
    if rally2['getpoint_player'][i] == 'A' or rally2['getpoint_player'][i] == 'B':
        balltype += [rally2['type'][i-1]]
        
result['balltype'] = balltype

result = (result.groupby(['set'], as_index=False)
            .apply(lambda x: x[['rally','score','stroke','winner','on_off_court','balltype','lose_area']].to_dict('r'))
            .reset_index()
            .rename(columns={0:'result'})
            )

export_json(filepath,result)

#Process detail data real
#### May need to modify file path and file name ####
filename = './clip_info_18IND_TC'
filepath = './Statistics/rally_detail_real_' + filename.split('_')[-1] + '.json'

rally = pd.read_excel(filename + '.xlsx')
rally = rally[['unique_id','rally','ball_round','getpoint_player','lose_reason','type','roundscore_A','roundscore_B','hit_x','hit_y']]

for i in range(len(rally['unique_id'])):
    rally['unique_id'][i] = rally['unique_id'][i].split('-')[-1]    

field = rally.groupby(['unique_id','rally'])

detail_idx = 0
score_idx = -1
result = pd.DataFrame(columns = ["set","rally","score","order","detail_type","detail_hit_pos"])
for i in range(len(field.groups.keys())):
    set = []
    rallys = []
    detail_type = []
    detail_hit_pos = []
    order = []
    score = []
    result_each = pd.DataFrame(columns = ["set","rally","score","order","detail_type","detail_hit_pos"])
    setrally = list(field.groups.keys())[i][0]
    hit_count = len(field.get_group(list(field.groups.keys())[i])['ball_round'].value_counts()) 
    score_idx += hit_count
    
    for j in range(hit_count):
        order += [j+1]
        detail_type += [rally['type'][detail_idx]]
        detail_hit_pos += [[rally['hit_x'][detail_idx],rally['hit_y'][detail_idx]]]
        set += [int(setrally)]
        rallys +=[list(field.groups.keys())[i][1]]
        if j == hit_count - 1:
            score_detail = str(rally['roundscore_A'][detail_idx-1]) + ':' + str(rally['roundscore_B'][detail_idx-1])
        else:
            score_detail = str(rally['roundscore_A'][detail_idx]) + ':' + str(rally['roundscore_B'][detail_idx])
        score_detail = str(rally['roundscore_A'][score_idx]) + ':' + str(rally['roundscore_B'][score_idx])
        score += [score_detail]
        detail_idx += 1
        
    result_each['set'] = set
    result_each['rally'] = rallys
    result_each['score'] = score
    result_each['order'] = order
    result_each['detail_type'] = detail_type
    result_each['detail_hit_pos'] = detail_hit_pos
    result = result.append(result_each)

result = (result.groupby(['set','rally','score'], as_index=False)
            .apply(lambda x: x[['order','detail_type','detail_hit_pos']].to_dict('r'))
            .reset_index()
            .rename(columns={0:'result'})
            )
result = (result.groupby(['set'], as_index=False)
            .apply(lambda x: x[['rally','score','result']].to_dict('r'))
            .reset_index()
            .rename(columns={0:'info'})
            )


export_json(filepath,result)