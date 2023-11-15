import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from tqdm import tqdm
from BC import BCAgent
import random


def preprocess():
    config = {}
    data = pd.read_csv('demo.csv')

    data = data[['rally_id', 'type', 'landing_x', 'landing_y',
                 'player_location_x', 'player_location_y', 'opponent_location_x', 'opponent_location_y']]
    mean_x, std_x = 175., 82.
    mean_y, std_y = 467., 192.
    data['landing_x'] /= std_x
    data['landing_y'] /= std_y
    data['player_location_x'] /= std_x
    data['player_location_y'] /= std_y
    data['opponent_location_x'] /= std_x
    data['opponent_location_y'] /= std_y

    grouped = data.groupby(['rally_id'])
    config['max_length'] = 2
    filtered = grouped.filter(lambda x: len(x) >= 3 and x['type'].iloc[1] != '發短球' and x['type'].iloc[1] != '發長球'
                              and (x['type'].iloc[0] == '發短球' or x['type'].iloc[0] == '發長球'))
    head = filtered.groupby(['rally_id']).head(config['max_length']+1)
    first_rows = head.groupby(['rally_id']).filter(
        lambda group: not group.isna().any().any())

    type_codes, type_uniques = pd.factorize(first_rows['type'])
    first_rows['type'] = type_codes + 1  # 0 is padding
    config['uniques_type'] = type_uniques.to_list()
    config['type_count'] = len(type_uniques) + 1

    # split training set and test set
    rally_index = first_rows['rally_id'].unique()
    train_num = int(len(rally_index) * 0.8)

    train_index = rally_index[:train_num]
    test_index = rally_index[train_num:]

    train_set = first_rows[first_rows['rally_id'].isin(
        train_index)].reset_index(drop=True)
    test_set = first_rows[first_rows['rally_id'].isin(
        test_index)].reset_index(drop=True)

    return train_set, test_set, config


def test(test_set: pd.DataFrame, predictor: BCAgent, config: dict):
    criterion = {
        'ce': nn.CrossEntropyLoss(ignore_index=0, reduction='sum'),
        'mse': nn.MSELoss(reduction='sum'),
        'mae': nn.L1Loss(reduction='sum')
    }

    loss = {
        'first_land_MSE': 0., 'first_land_MAE': 0.,
        'first_move_MSE': 0., 'first_move_MAE': 0.,
        'first_shot_CE': 0.,
        'second_land_MSE': 0., 'second_land_MAE': 0.,
        'second_move_MSE': 0., 'second_move_MAE': 0.,
        'second_shot_CE': 0.
    }

    max_length = config['max_length']

    grouped = test_set.groupby(['rally_id'])
    count = len(grouped)
    for group_name, rows in tqdm(grouped):
        player_x = [0] * max_length
        player_y = [0] * max_length
        opponent_x = [0] * max_length
        opponent_y = [0] * max_length
        ball_x = [0] * max_length
        ball_y = [0] * max_length
        prev_type = [0] * max_length
        for i in range(0, len(rows)-1):
            next_row = rows.iloc[i+1]
            row = rows.iloc[i]

            # state
            player_x[i] = row['player_location_x']
            player_y[i] = row['player_location_y']
            opponent_x[i] = row['opponent_location_x']
            opponent_y[i] = row['opponent_location_y']
            if i == 0:
                prev_type[i] = 0
                ball_x[i] = player_x[i]
                ball_y[i] = player_y[i]
            else:
                last_row = rows.iloc[i-1]
                prev_type[i] = last_row['type']
                ball_x[i] = -last_row['landing_x']
                ball_y[i] = -last_row['landing_y']
            player = torch.tensor([[player_x, player_y]])
            opponent = torch.tensor([[opponent_x, opponent_y]])
            ball = torch.tensor([[ball_x, ball_y]])
            type = torch.tensor([prev_type]).long()

            predict = predictor.predict(player, opponent, ball, type)
            predict_shot, predict_land, predict_move = predict

            prev_player_coord = opponent * -1

            # action
            landing_x = row['landing_x']
            landing_y = row['landing_y']
            move_x = -next_row['opponent_location_x']
            move_y = -next_row['opponent_location_y']
            shot = torch.tensor([row['type']]).long()
            land = torch.tensor([[landing_x, landing_y]])
            move = torch.tensor([[move_x, move_y]])

            shot_CE = criterion['ce'](predict_shot, shot).item()
            land_MSE = criterion['mse'](predict_land, land).item()
            move_MSE = criterion['mse'](predict_move, move).item()
            land_MAE = criterion['mae'](predict_land, land).item()
            move_MAE = criterion['mae'](predict_move, move).item()

            if i == 0:
                round = 'first'
            elif i == 1:
                round = 'second'
            else:
                raise NotImplementedError

            loss[f'{round}_shot_CE'] += shot_CE
            loss[f'{round}_land_MSE'] += land_MSE
            loss[f'{round}_move_MSE'] += move_MSE
            loss[f'{round}_land_MAE'] += land_MAE
            loss[f'{round}_move_MAE'] += move_MAE

    for key in loss.keys():
        loss[key] /= count

    return loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # gpu vars
    torch.use_deterministic_algorithms(True)


def main():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    training_set, testing_set, config = preprocess()
    # config only for training
    print(config)

    set_seed(1)

    predictor = BCAgent(True, training_set, config)
    predictor.learn()

    filename = f'ShuttleDyMFNet2ball-final.pt'
    predictor.save(filename)
    # predictor.load(filename)

    loss = test(testing_set, predictor, config)

    print(f'iteration:100000' +
          'first: [shot CE: {first_shot_CE:.4f}] '
          '[land MSE: {first_land_MSE:.4f}] '
          '[land MAE: {first_land_MAE:.4f}] '
          '[move MSE: {first_move_MSE:.4f}] '
          '[move MAE: {first_move_MAE:.4f}]\n'
          'second: [shot CE: {second_shot_CE:.4f}] '
          '[land MSE: {second_land_MSE:.4f}] '
          '[land MAE: {second_land_MAE:.4f}] '
          '[move MSE: {second_move_MSE:.4f}] '
          '[move MAE: {second_move_MAE:.4f}]'.format(**loss))


if __name__ == '__main__':
    main()
