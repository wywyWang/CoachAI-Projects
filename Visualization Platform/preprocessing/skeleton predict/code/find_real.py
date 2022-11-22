import pandas as pd
import matplotlib.pyplot as plt
import math
import json
import numpy as np
import cv2

game_name = '18ENG_TC'
ext = '.csv'

bot_filename = '../data/'+str(game_name)+'/player_skeleton/bottom_player_skeleton'
top_filename = '../data/'+str(game_name)+'/player_skeleton/top_player_skeleton'

raw_bot_filename = bot_filename+str(ext)
raw_top_filename = top_filename+str(ext)

fill_bot = bot_filename+'_fill'+str(ext)
fill_top = top_filename+'_fill'+str(ext)

def save_to_csv(sets, frame, top_left_x, top_left_y, top_right_x, top_right_y, bot_left_x, bot_left_y, bot_right_x, bot_right_y):
    result = pd.DataFrame([])
    result['frame'] = frame
    result['top_right_x'] = top_right_x
    result['top_right_y'] = top_right_y
    result['top_left_x'] = top_left_x
    result['top_left_y'] = top_left_y
    
    result['bot_right_x'] = top_right_x
    result['bot_right_y'] = top_right_y
    result['bot_left_x'] = bot_left_x
    result['bot_left_y'] = bot_left_y

    result.to_csv("../data/"+str(game_name)+"/player_skeleton/"+str(game_name)+"_set"+str(sets+1)+"_skeleton"+str(ext), index=False, encoding = 'utf-8')

def find_real():
    player_bot = pd.read_csv(fill_bot)
    player_top = pd.read_csv(fill_top)

    # 真實世界球場位置
    dst=np.array([[610,1340],[0,1340],[610,0],[0,0]], np.float32)

    # 影像中球場位置(右下、左下、右上、左上): !!! 要重新確認座標點 !!!
    src=np.array([[1011,671],[276,671],[880,383],[404,383]], np.float32)

    H = cv2.getPerspectiveTransform(src,dst)
    frame = []
    top_left_x = []
    top_left_y = []
    top_right_x = []
    top_right_y = []
    bot_left_x = []
    bot_left_y = []
    bot_right_x = []
    bot_right_y = []
    sets = 0

    for i in range(len(player_bot)):
        #img = draw_court()
        
        if sets != int(player_bot['set'][i]):
            save_to_csv(sets, frame, top_left_x, top_left_y, top_right_x, top_right_y, bot_left_x, bot_left_y, bot_right_x, bot_right_y)
            frame = []
            top_left_x = []
            top_left_y = []
            top_right_x = []
            top_right_y = []
            bot_left_x = []
            bot_left_y = []
            bot_right_x = []
            bot_right_y = []
            
            sets = int(player_bot['set'][i])
        
        frame.append(player_bot['frame_id'][i])

        # middle point of bottom player
        bot_mid_x = (player_bot['x11'][i] + player_bot['x14'][i])/2
        bot_mid_y = (player_bot['y11'][i] + player_bot['y14'][i])/2
        bot_mid_point = np.array([bot_mid_x,bot_mid_y,1])
        bot_H_point = H.dot(bot_mid_point.transpose()).round(3)
        bot_H_point /= bot_H_point[2]
        bot_H_point = bot_H_point.astype(int)
        bot_point = tuple([bot_H_point[0],bot_H_point[1]])
        
        # middle point of top player
        top_mid_x = (player_top['x11'][i] + player_top['x14'][i])/2
        top_mid_y = (player_top['y11'][i] + player_top['y14'][i])/2
        top_mid_point = np.array([top_mid_x,top_mid_y,1])
        top_H_point = H.dot(top_mid_point.transpose()).round(3)
        top_H_point /= top_H_point[2]
        top_H_point = top_H_point.astype(int)
        top_point = tuple([top_H_point[0],top_H_point[1]])
        
        # right ankle and left ankle of bottom player
        bot_right_ankle = np.array([player_bot['x11'][i],player_bot['y11'][i],1])
        bot_H_right_ankle = H.dot(bot_right_ankle.transpose()).round(3)
        bot_H_right_ankle /= bot_H_right_ankle[2]
        bot_H_right_ankle = bot_H_right_ankle.astype(int)
        bot_H_right_ankle = tuple([bot_H_right_ankle[0],bot_H_right_ankle[1]])
        bot_left_ankle = np.array([player_bot['x14'][i],player_bot['y14'][i],1])
        bot_H_left_ankle = H.dot(bot_left_ankle.transpose()).round(3)
        bot_H_left_ankle /= bot_H_left_ankle[2]
        bot_H_left_ankle = bot_H_left_ankle.astype(int)
        bot_H_left_ankle = tuple([bot_H_left_ankle[0],bot_H_left_ankle[1]])
        
        # right ankle and left ankle of top player
        top_right_ankle = np.array([player_top['x11'][i],player_top['y11'][i],1])
        top_H_right_ankle = H.dot(top_right_ankle.transpose()).round(3)
        top_H_right_ankle /= top_H_right_ankle[2]
        top_H_right_ankle = top_H_right_ankle.astype(int)
        top_H_right_ankle = tuple([top_H_right_ankle[0],top_H_right_ankle[1]])
        top_left_ankle = np.array([player_top['x14'][i],player_top['y14'][i],1])
        top_H_left_ankle = H.dot(top_left_ankle.transpose()).round(3)
        top_H_left_ankle /= top_H_left_ankle[2]
        top_H_left_ankle = top_H_left_ankle.astype(int)
        top_H_left_ankle = tuple([top_H_left_ankle[0],top_H_left_ankle[1]])

        bot_right_x.append(bot_H_right_ankle[0] if bot_H_right_ankle[0] != -2147483648 else 0)
        bot_right_y.append(bot_H_right_ankle[1] if bot_H_right_ankle[1] != -2147483648 else 0)
        bot_left_x.append(bot_H_left_ankle[0] if bot_H_left_ankle[0] != -2147483648 else 0)
        bot_left_y.append(bot_H_left_ankle[1] if bot_H_left_ankle[1] != -2147483648 else 0)

        top_right_x.append(top_H_right_ankle[0] if top_H_right_ankle[0] != -2147483648 else 0)
        top_right_y.append(top_H_right_ankle[1] if top_H_right_ankle[1] != -2147483648 else 0)
        top_left_x.append(top_H_left_ankle[0] if top_H_left_ankle[0] != -2147483648 else 0)
        top_left_y.append(top_H_left_ankle[1] if top_H_left_ankle[1] != -2147483648 else 0)

    save_to_csv(sets, frame, top_left_x, top_left_y, top_right_x, top_right_y, bot_left_x, bot_left_y, bot_right_x, bot_right_y)

def check_empty(data):
    empty_idx = []
    empty_cnt = []
    is_empty = False
    cnt = 0
    total_empty = 0
    for i in range(len(data)):
        # data[i] != data[i] => check nan
        if data[i] != data[i] and not is_empty:
            is_empty = True
            empty_idx.append(i)
        elif data[i] == data[i] and is_empty:
            is_empty = False
            empty_cnt.append(cnt)
            cnt = 0

        if is_empty:
            cnt += 1
            total_empty+=1

    if cnt:
        empty_cnt.append(cnt)
    print("total: "+str(len(data))+" empty: "+str(total_empty))
    return empty_idx, empty_cnt

def fill(data, empty_idx, empty_cnt):

    for i in range(len(empty_idx)):
        # empty at index 0
        if empty_idx[i] == 0:
            for j in range(empty_cnt[i]):
                data.loc[j, ['x11']]= data['x11'][empty_cnt[i]]-(j+1)*((j+1)%2)
                data.loc[j, ['x14']]= data['x14'][empty_cnt[i]]-(j+1)*((j+1)%2)
                data.loc[j, ['y11']] = data['y11'][empty_cnt[i]]-(j+1)*((j+1)%2)
                data.loc[j, ['y14']] = data['y14'][empty_cnt[i]]-(j+1)*((j+1)%2)
        # empty until the last of data
        elif empty_idx[i]+empty_cnt[i] == len(data):
            for j in range(empty_idx[i], len(data)):
                data.loc[j, ['x11']] = data['x11'][empty_idx[i]-1]-(j+1)*((j+1)%2)
                data.loc[j, ['x14']] = data['x14'][empty_idx[i]-1]-(j+1)*((j+1)%2)
                data.loc[j, ['y11']] = data['y11'][empty_idx[i]-1]-(j+1)*((j+1)%2)
                data.loc[j, ['y14']] = data['y14'][empty_idx[i]-1]-(j+1)*((j+1)%2)
        # empty in middle
        else:
            start_x11 = data['x11'][empty_idx[i]-1]
            start_x14 = data['x14'][empty_idx[i]-1]
            start_y11 = data['y11'][empty_idx[i]-1]
            start_y14 = data['y14'][empty_idx[i]-1]

            end_x11 = data['x11'][empty_idx[i]+empty_cnt[i]]
            end_x14 = data['x14'][empty_idx[i]+empty_cnt[i]]
            end_y11 = data['y11'][empty_idx[i]+empty_cnt[i]]
            end_y14 = data['y14'][empty_idx[i]+empty_cnt[i]]

            dif_x11 = (end_x11-start_x11)/empty_cnt[i]
            dif_x14 = (end_x14-start_x14)/empty_cnt[i]
            dif_y11 = (end_y11-start_y11)/empty_cnt[i]
            dif_y14 = (end_y14-start_y14)/empty_cnt[i]

            for j in range(empty_cnt[i]):
                data.loc[empty_idx[i]+j, ['x11']] = dif_x11*(j+1)+start_x11
                data.loc[empty_idx[i]+j, ['x14']] = dif_x14*(j+1)+start_x14
                data.loc[empty_idx[i]+j, ['y11']] = dif_y11*(j+1)+start_y11
                data.loc[empty_idx[i]+j, ['y14']] = dif_y14*(j+1)+start_y14

    return data

def fill_empty():
    data_top = pd.read_csv(raw_top_filename)
    data_bot = pd.read_csv(raw_bot_filename)
    #total: 18242 empty: 4871
    empty_top_idx, empty_top_cnt = check_empty(data_top['x11'])
    #total: 18242 empty: 527
    empty_bot_idx, empty_bot_cnt = check_empty(data_bot['x11'])
    player_top = fill(data_top, empty_top_idx, empty_top_cnt)
    player_bot = fill(data_bot, empty_bot_idx, empty_bot_cnt)
    player_top.to_csv(fill_top, index=False, encoding = 'utf-8')
    player_bot.to_csv(fill_bot, index=False, encoding = 'utf-8')

def exec():
    fill_empty()
    find_real()

exec()