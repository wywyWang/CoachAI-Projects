import torch
import numpy as np
import pandas as pd
import random
from data_cleaner import DataCleaner
from dataset import BadmintonDataset

from torch.utils.data import DataLoader

def prepare_dataset(args):
    matches = DataCleaner(args)
    
    used_column = ['rally_id', 'player', 'type', 'player_location_x', 'player_location_y', 'opponent_location_x', 'opponent_location_y', 'ball_round', 'set', 'match_id']
    matches = matches[used_column]

    player_codes, player_uniques = pd.factorize(matches['player'])
    matches['player'] = player_codes + 1
    print(matches['player'])
    args['player_num'] = len(player_uniques) + 1

    type_codes, type_uniques = pd.factorize(matches['type'])
    matches['type'] = type_codes + 1
    args['type_num'] = len(type_uniques) + 1
    
    train_index = []
    valid_index = []
    test_index = []

    for match_id in matches['match_id'].unique():
        match = matches[matches['match_id']==match_id]
        rally_index = match['rally_id'].unique()

        train_num = int(len(rally_index) * args['train_ratio'])
        valid_num = int(len(rally_index) * args['valid_ratio'])

        train_index.extend(rally_index[:train_num])
        valid_index.extend(rally_index[train_num:train_num+valid_num])
        test_index.extend(rally_index[train_num+valid_num:])
    
    train_index = np.array(train_index)
    valid_index = np.array(valid_index)
    test_index = np.array(test_index)

    train_rally_data = matches[matches['rally_id'].isin(train_index)].reset_index(drop=True)
    valid_rally_data = matches[matches['rally_id'].isin(valid_index)].reset_index(drop=True)
    test_rally_data = matches[matches['rally_id'].isin(test_index)].reset_index(drop=True)

    train_dataset = BadmintonDataset(train_rally_data, used_column, args)
    valid_dataset = BadmintonDataset(valid_rally_data, used_column, args)
    test_dataset = BadmintonDataset(test_rally_data, used_column, args)

    g = torch.Generator()
    g.manual_seed(0)

    train_dataloader = DataLoader(train_dataset, batch_size=args['train_batch_size'], shuffle=True, num_workers=8)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args['valid_batch_size'], shuffle=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=args['test_batch_size'], shuffle=False, num_workers=8)
    return train_dataloader, valid_dataloader, test_dataloader, args
    
def prepare_predict_dataset1(args):
    matches = DataCleaner(args)
    
    used_column = ['rally_id', 'player', 'type', 'player_location_x', 'player_location_y', 'opponent_location_x', 'opponent_location_y', 'ball_round', 'set', 'match_id']
    matches = matches[used_column]

    player_codes, player_uniques = pd.factorize(matches['player'])
    matches['player'] = player_codes + 1
    print(len(matches))
    #args['player_num'] = len(player_uniques) + 1

    type_codes, type_uniques = pd.factorize(matches['type'])
    matches['type'] = type_codes + 1
    #args['type_num'] = len(type_uniques) + 1
    
    
    dataset = BadmintonDataset(matches, used_column, args)

    g = torch.Generator()
    g.manual_seed(0)
    
    #dataloader = DataLoader(dataset, batch_size=args['test_batch_size'], shuffle=False, num_workers=8)
    return dataset, args

def prepare_predict_dataset(args):
    matches = pd.read_csv(args['preprocessed_data_path'])
    
    used_column = ['rally_id', 'player', 'type', 'player_location_x', 'player_location_y', 'opponent_location_x', 'opponent_location_y', 'ball_round', 'set', 'match_id']
    matches = matches[used_column]

    player_codes, player_uniques = pd.factorize(matches['player'])
    matches['player'] = player_codes + 1
    print(len(matches))
    #args['player_num'] = len(player_uniques) + 1

    #type_codes, type_uniques = pd.factorize(matches['type'])
    #matches['type'] = type_codes + 1
    #args['type_num'] = len(type_uniques) + 1
    for i, uniquesType in enumerate(args['uniques_type']):
        matches.loc[matches['type']==uniquesType,'type'] = i + 1
    matches['type'] = matches['type'].astype(int)
    
    # dataset = BadmintonDataset(matches, used_column, args)

    player_sequence = []

    encode_length = args['max_length']
    rally_id_grouped = matches.groupby('rally_id').groups

    rally_data = matches[used_column]        

    for rally_id in rally_id_grouped.values():
        rally_id = rally_id.to_numpy()
        one_rally = rally_data.iloc[rally_id].reset_index(drop=True)
        
        if len(one_rally) > encode_length:
            seqence_length = encode_length
            one_rally = one_rally.head(encode_length)
        else:
            seqence_length = len(one_rally)
            
        # rally information
        player = one_rally[['player']].values[:].reshape(-1)
        shot_type = one_rally[['type']].values[:].reshape(-1)

        player_x = one_rally[['player_location_x']].values[:].reshape(-1)
        player_y = one_rally[['player_location_y']].values[:].reshape(-1)
        opponent_x = one_rally[['opponent_location_x']].values[:].reshape(-1)
        opponent_y = one_rally[['opponent_location_y']].values[:].reshape(-1)

        player = np.pad(player, (0, encode_length - seqence_length), 'constant', constant_values=(0))
        shot_type  = np.pad(shot_type , (0, encode_length - seqence_length), 'constant', constant_values=(0))
        player_x = np.pad(player_x, (0, encode_length - seqence_length), 'constant', constant_values=(0))
        player_y = np.pad(player_y, (0, encode_length - seqence_length), 'constant', constant_values=(0))
        opponent_x = np.pad(opponent_x, (0, encode_length - seqence_length), 'constant', constant_values=(0))
        opponent_y = np.pad(opponent_y, (0, encode_length - seqence_length), 'constant', constant_values=(0))           
        
        player_A_x = np.empty((args['max_length'],), dtype=float)
        player_A_x[0::2] = player_x[0::2]
        player_A_x[1::2] = opponent_x[1::2]
        player_A_y = np.empty((args['max_length'],), dtype=float)
        player_A_y[0::2] = player_y[0::2]
        player_A_y[1::2] = opponent_y[1::2]

        player_B_x = np.empty((args['max_length'],), dtype=float)
        player_B_x[0::2] = opponent_x[0::2]
        player_B_x[1::2] = player_x[1::2]
        player_B_y = np.empty((args['max_length'],), dtype=float)
        player_B_y[0::2] = opponent_y[0::2]
        player_B_y[1::2] = player_y[1::2]

        player_sequence.append([player, shot_type, player_A_x, player_A_y, player_B_x, player_B_y])

    g = torch.Generator()
    g.manual_seed(0)
    
    #dataloader = DataLoader(dataset, batch_size=args['test_batch_size'], shuffle=False, num_workers=8)
    return player_sequence, args