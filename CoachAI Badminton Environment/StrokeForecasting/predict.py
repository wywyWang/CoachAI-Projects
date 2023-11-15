from badmintoncleaner import prepare_testDataset
from utils import predict
import ast
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # gpu vars

mid_point = [177.5, 480]


def main(dataSetPath = './StrokeForecasting/data/datasetTest.csv'):
    #model_path = sys.argv[1]
    model_path = "./StrokeForecasting/discreteFinal"
    
    config = ast.literal_eval(open(f"{model_path}1/config", encoding='utf8').readline())
    SAMPLES = 50  # config['sample']
    set_seed(config['seed_value'])

    print(config['uniques_type'])
    # Prepare Dataset
    dataSet = pd.read_csv(dataSetPath)

    results = [] #[x, y, shotType, player, sx, sy, tho]
    for rally, group in dataSet.groupby(['rally']):

        matches, testData, config = prepare_testDataset(
            config, dataSet[dataSet['rally'] == rally])
        #print(f"palyer count{config['player_num']}")

        # if the input ball length is shorter than 4
        # output directly since model need 4 ball to predict
        if len(testData[0]) < 4:
            shot_type, landing_x, landing_y, player = testData
            result = []
            for x, y, shot, p in zip(landing_x, landing_y, shot_type, player):
                shotTypeName = config['uniques_type'][shot-1]
                result.append((x, y, shotTypeName, p, 0.00001, 0.00001, 0, []))
            results.append((rally, result))
            continue

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load model
        from ShuttleNet.Decoder import ShotGenDecoder
        from ShuttleNet.Encoder import ShotGenEncoder
        encoder = ShotGenEncoder(config)
        decoder = ShotGenDecoder(config)

        encoder.to(device), decoder.to(device)
        current_model_path = f"{model_path}1/"
        encoder_path = f"{current_model_path}encoder"
        decoder_path = f"{current_model_path}decoder"
        encoder.load_state_dict(torch.load(encoder_path)), decoder.load_state_dict(
            torch.load(decoder_path))

        total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) \
            + sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        #print(f"Model params: {total_params}")

        # run prediction
        result = predict(testData, encoder, decoder, config,
                         samples=SAMPLES, device=device)
        results.append((rally, result))
        # print(result)

    shot = pd.DataFrame()

    for rally, result in results:
        currentBallRound = 1
        for (x, y, shotType, player, sx, sy, tho, shot_prob) in result:
            row = pd.DataFrame([{'rally': rally,
                                 'ball_round': currentBallRound,
                                 'player': player,
                                 'type': shotType,
                                 'landing_x': x,
                                 'landing_y': y,
                                 'landing_sx': sx,
                                 'landing_sy': sy,
                                 'landing_tho': tho,
                                 'shot_prob': shot_prob}])
            shot = pd.concat([shot, row], ignore_index=True)
            currentBallRound += 1

    # convert standardize coord back to normal coord
    mean_x, std_x = 175., 82.
    mean_y, std_y = 467., 192.
    shot['landing_x'] = shot['landing_x'] * std_x + mean_x
    shot['landing_y'] = shot['landing_y'] * std_y + mean_y
    shot['landing_sx'] = shot['landing_sx'] *std_x
    shot['landing_sy'] = shot['landing_sy'] *std_y
    shot.to_csv('predict_ShuttleNet.csv', index= False)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        main()
