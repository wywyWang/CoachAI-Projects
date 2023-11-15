import sys
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import pickle

from prepare_dataset import prepare_predict_dataset, prepare_dataset

def load_args_file(model_folder):
    with open(model_folder + '/config', 'rb') as f:
        args = pickle.load(f)
    return args

import os

def main(data_path = './data/datasetTest.csv'):
    #model_folder = sys.argv[1]
    #sample_num = sys.argv[2]

    args = load_args_file("./model/DyMF_2_20230826")#model_folder
    args['sample_num'] = args['max_length']#int(sample_num)

    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args['uniques_type'] = ['發短球', '長球', '推撲球', '殺球', '接殺防守', '平球', '網前球', '挑球', '切球', '發長球']


    args['already_have_data'] = True
    args['preprocessed_data_path'] = data_path 
    datas, args = prepare_predict_dataset(args)
    matches = pd.read_csv(args['preprocessed_data_path'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args['model_type'] == 'DyMF':
        from DyMF.model import Encoder, Decoder
        from DyMF.runner import predict

        encoder = Encoder(args, device)
        decoder = Decoder(args, device)
        encoder.player_embedding.weight = decoder.player_embedding.weight
        encoder.coordination_transform.weight = decoder.coordination_transform.weight
        # encoder.rGCN.type_embedding.weight = decoder.rGCN.type_embedding.weight

    else:
        raise NotImplementedError

    try:
        encoder.load_state_dict(torch.load(args['model_folder'] + '/encoder')), decoder.load_state_dict(torch.load(args['model_folder'] + '/decoder'))
    except:
        encoder.load_state_dict(torch.load('MovementForecasting/model/DyMF_4_20230120/encoder')), decoder.load_state_dict(torch.load('MovementForecasting/model/DyMF_4_20230120/decoder'))

    encoder.to(device), decoder.to(device)

    outputs = []
    for data in datas: # data = (player, shot_type, player_A_x, player_A_y, player_B_x, player_B_y)
        for i in range(len(data)):
            data[i] = np.trim_zeros(data[i])

        if len(data[0]) < args['encode_length']:
            outputs.append(data)
        else:
            player, shot_type, A_x, A_y, B_x, B_y = predict(data, encoder, decoder, args, samples = 50, device=device)
            outputs.append([player, shot_type, A_x, A_y, B_x, B_y])

        if outputs[-1][3][0] < 0:  #A_y[0]
            outputs[-1][2], outputs[-1][4] = outputs[-1][4], outputs[-1][2]
            outputs[-1][3], outputs[-1][5] = outputs[-1][5], outputs[-1][3]

    mean_x, std_x = 175., 82.
    mean_y, std_y = 467., 192.


    shot = pd.DataFrame()
    for output, group in zip(outputs,matches.groupby(['rally'])):
        rally = group[0]
        ballround = 1
        for (player, shot_type, player_A_x, player_A_y, player_B_x ,player_B_y) in \
            zip(output[0], output[1], output[2], output[3], output[4], output[5]):
            row = pd.DataFrame([{'rally': rally,
                                 'ball_round': ballround,
                                 'player': player,
                                 'type': args['uniques_type'][shot_type-1],
                                 'player_A_x': player_A_x,
                                 'player_A_y': player_A_y,
                                 'player_B_x': player_B_x,
                                 'player_B_y': player_B_y}])
            shot = pd.concat([shot, row], ignore_index=True)
            ballround += 1

    shot['player_A_x'] = shot['player_A_x'] * std_x + mean_x # player_A_x
    shot['player_A_y'] = 960 - (shot['player_A_y'] * std_y + mean_y) # player_A_x
    shot['player_B_x'] = shot['player_B_x'] * std_x + mean_x # player_A_x
    shot['player_B_y'] = 960 - (shot['player_B_y'] * std_y + mean_y) # player_A_x

    shot.to_csv('predict_movement.csv', index= False)

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        main()