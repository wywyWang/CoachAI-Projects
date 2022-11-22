import numpy as np
import pandas as pd
import math
import functions

def has_frame(data, target):
    for i in range(len(data)):
        if data["frame_num"][i] == target:
            return True, i
    return False, -1

def get_velocity(filename, unique_id, savename):
    data = pd.read_csv(filename, encoding = 'utf-8')

    # extract columns
    hit_x = data['hit_x']
    hit_y = data['hit_y']
    hit_area = data['hit_area']
    landing_x = data['landing_x']
    landing_y = data['landing_y']
    landing_area = data['landing_area']
    time = data['time']

    lose_reason = data['lose_reason']
    ball_type = data['type']
    
    data_num = data.shape[0]

    velocity = []
    direction = []

    for i in range(data_num):
        
        if type(lose_reason[i]) != float:
            velocity.append('0')
            direction.append('')
            continue
        
        # velocity
        v = functions.velocity(time[i],time[i+1],hit_x[i],hit_y[i],landing_x[i],landing_y[i])
        velocity.append(v)

        # direction (first para means diagonal angle)
        d = functions.direction(30,hit_x[i],hit_y[i],hit_area[i],landing_x[i],landing_y[i],landing_area[i])
        direction.append(d)

    data['velocity'] = velocity
    data['direction'] = direction
    data.insert(loc=0, column='unique_id', value=unique_id)
    data.to_csv(savename,index=False, encoding='utf-8')
        
def process(filename, clipinfo, player_pos_option, frame_option, player_pos_file, specific_frame_file):
    shift = 565

    ball_type_new = []
    hit_direct = []
    hit_distance = []
    hit_height_new = []
    landing_direct = []
    landing_distance = []
    landing_height_new = []
    x_direct = []
    x_distance = []
    y_direct = []
    y_distance = []

    data = pd.read_csv(filename, encoding = 'utf-8')
    clip = pd.read_excel(clipinfo, encoding = 'utf-8')

    for f in data["frame_num"]:
        delta = [-2, -1, 0, 1, 2]
        find = False
        for d in delta:
            if int(f)+int(d)+shift > 0 and has_frame(clip, int(f)+int(d)+shift)[0]:
                ball_type_new.append(functions.ball_type_convertion(clip["type"][has_frame(clip, int(f)+int(d)+shift)[1]]))
                find = True
                break
        if not find:
            ball_type_new.append("error")

    data_num = data.shape[0]

    frame = data["frame_num"]
    #ball_type = data["type"]

    hit_area = data["hit_area"]
    #hit_height = data["hit_height"]
    lose_reason = data["lose_reason"]
    landing_area = data["landing_area"]
    landing_x = data["landing_x"]
    landing_y = data["landing_y"]
    #landing_height = data["landing_height"]
    velocity = data["velocity"]

    need_frame = 0
    data_num = data.shape[0]

    # If there is player position info
    if player_pos_option == 1:
        df = pd.read_csv(player_pos_file)
        data_num = need_frame.shape[0]

    # Decide if trained with the specific frame
    if frame_option == 1:
        df = pd.read_csv(specific_frame_file)
        need_frame = df.values
        data_num = need_frame.shape[0]

    # start
    for j in range(data_num):
        if frame_option == 1:
            result = np.where(frame == need_frame[j])
            i = result[0][0]
        else:
            i = j

        #if type(lose_reason[i]) != float or type(hit_area[i]) == float or type(ball_type[i]) == float:
        #    continue
        if type(lose_reason[i]) != float or type(hit_area[i]) == float:
            continue

        if player_pos_option == 1:
            # x direct & x distance
            x_dir , x_dis = functions.hit_convertion_9(hit_area[i])
            x_direct.append(x_dir)
            x_distance.append(x_dis)

            # y direct & y distance
            y_dir , y_dis = functions.landing_convertion_9(hit_area[i])
            y_direct.append(y_dir)
            y_distance.append(y_dis)
        else:
            x_direct.append('0')
            x_distance.append('0')
            y_direct.append('0')
            y_distance.append('0')

        # hit direct & hit distance
        h_dir , h_dis = functions.hit_convertion_9(hit_area[i])
        hit_direct.append(h_dir)
        hit_distance.append(h_dis)
        
        if h_dir == 'X':
            print('error hit_area: ',i,hit_area[i])
        
        # landing direct & landing distance
        d_dir , d_dis = functions.landing_convertion_9(landing_area[i])
        landing_direct.append(d_dir)
        landing_distance.append(d_dis)

        if d_dir == 'X':
            print('error landing_area: ',i,landing_area[i])

        # height
        #hit_height_new.append(int(hit_height[i]))
        #landing_height_new.append(int(landing_height[i]))
        hit_height_new.append('0')
        landing_height_new.append('0')

    output_data = pd.DataFrame([]) 
    output_data["hit_direct"] = hit_direct
    output_data["hit_distance"] = hit_distance
    output_data["hit_height"] = hit_height_new
    output_data["landing_direct"] = landing_direct
    output_data["landing_distance"] = landing_distance
    output_data["landing_height"] = landing_height_new

    output_data["x_direct"] = x_direct
    output_data["x_distance"] = x_distance
    output_data["y_direct"] = y_direct
    output_data["y_distance"] = y_distance

    output_data["velocity"] = velocity
    output_data["ball_type"] = pd.Series(ball_type_new)

    output_data.to_csv(filename, index=False, encoding = 'utf-8')

def run(filename, savename, clipinfo, unique_id, player_pos_option, frame_option, player_pos_file, specific_frame_file):
    print("Getting velocity...")
    get_velocity(filename, unique_id, savename)
    print("Getting velocity done...")
    print("")
    print("Starting main...")
    process(savename, clipinfo, player_pos_option, frame_option, player_pos_file, specific_frame_file)
    print("All done...")

def exec(game_names):
    for g in game_names:
        run("../../Data/training/data/record_segmentation_"+str(g)+"_out.csv", "../../Data/training/data/"+str(g)+"_preprocessed.csv", "../../Data/TrainTest/clip_info_"+str(g)+".xlsx",'', 0, 0, '', '')

exec(["19ASI_CS_10min"])