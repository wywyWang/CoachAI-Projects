from badmintondataset import BadmintonDataset
from torch.utils.data import DataLoader
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import ast
import os
import re
from sklearn.model_selection import KFold


class BadmintonCleaner:
    def __init__(self, directory, match_list):
        available_matches = pd.read_csv(directory + match_list)
        self.homography_matrix = pd.read_csv(directory + 'homography_new.csv', converters={'homography_matrix':lambda x: np.array(ast.literal_eval(x))})
        all_matches = self.read_match(directory, available_matches)
        cleaned_matches = self.engineer_match(all_matches)
        cleaned_matches.to_csv('data/dataset.csv', index=False)
    
    def read_match(self, directory, available_matches):
        """Read all matches and concat to one dataframe

        Args:
            directory (string): Base folder of all matches
            available_matches (DataFrame): List of matches read from match.csv

        Returns:
            DataFrame: all sets of matches concatenation
        """
        all_matches = []
        for idx in range(len(available_matches)):
            match_idx = available_matches['id'][idx]
            match_name = available_matches['video'][idx]
            winner = available_matches['winner'][idx]
            loser = available_matches['loser'][idx]
            current_homography = self.homography_matrix[self.homography_matrix['id'] == match_idx]['homography_matrix'].to_numpy()[0]

            match_path = os.path.join(directory, match_name)
            csv_paths = [os.path.join(match_path, f) for f in os.listdir(match_path) if f.endswith('.csv')]

            one_match = []
            for csv_path in csv_paths:
                data = pd.read_csv(csv_path)
                data['player'] = data['player'].replace(['A', 'B'], [winner, loser])
                data['set'] = re.findall(r'\d+', os.path.basename(csv_path))[0]
                one_match.append(data)

            match = pd.concat(one_match, ignore_index=True, sort=False).assign(match_id=match_idx)
            
            # project screen coordinate to real coordinate
            for i in range(len(match)):
                p = np.array([match['landing_x'][i], match['landing_y'][i], 1])
                # p = np.array([407.6, 308.6, 1])        # for test -> (27.40, 150))
                p_real = current_homography.dot(p)
                p_real /= p_real[2]
                match['landing_x'][i], match['landing_y'][i] = p_real[0], p_real[1]

            all_matches.append(match)

        all_matches = pd.concat(all_matches, ignore_index=True, sort=False)
        return all_matches

    def engineer_match(self, matches):
        matches['rally_id'] = matches.groupby(['match_id', 'set', 'rally']).ngroup()
        print("Original: ")
        self.print_current_size(matches)

        # Drop flaw rally
        if 'flaw' in matches.columns:
            flaw_rally = matches[matches['flaw'].notna()]['rally_id']
            matches = matches[~matches['rally_id'].isin(flaw_rally)]
            matches = matches.reset_index(drop=True)
        print("After Dropping flaw: ")
        self.print_current_size(matches)

        # Drop unknown ball type
        unknown_rally = matches[matches['type'] == '未知球種']['rally_id']
        matches = matches[~matches['rally_id'].isin(unknown_rally)]
        matches = matches.reset_index(drop=True)
        print("After dropping unknown ball type: ")
        self.print_current_size(matches)

        # Drop hit_area at outside
        outside_area = [10, 11, 12, 13, 14, 15, 16]
        matches.loc[matches['server'] == 1, 'hit_area'] = 7
        for area in outside_area:
            outside_rallies = matches.loc[matches['hit_area'] == area, 'rally_id']
            matches = matches[~matches['rally_id'].isin(outside_rallies)]
            matches = matches.reset_index(drop=True)
        # Deal with hit_area convert hit_area to integer
        matches = self.drop_na_rally(matches, columns=['hit_area'])
        matches['hit_area'] = matches['hit_area'].astype(float).astype(int)
        print("After converting hit_area: ")
        self.print_current_size(matches)

        # Convert landing_area outside to 10 and to integer
        matches = self.drop_na_rally(matches, columns=['landing_area'])
        for area in outside_area:
            matches.loc[matches['landing_area'] == area, 'landing_area'] = 10
        matches['landing_area'] = matches['landing_area'].astype(float).astype(int)
        print("After converting landing_area: ")
        self.print_current_size(matches)

        # Deal with ball type. Convert ball types to general version (10 types)
        # Convert 小平球 to 平球 because of old version
        matches['type'] = matches['type'].replace('小平球', '平球')
        combined_types = {'切球': '切球', '過度切球': '切球', '點扣': '殺球', '殺球': '殺球', '平球': '平球', '後場抽平球': '平球', '擋小球': '接殺防守',
                    '防守回挑': '接殺防守', '防守回抽': '接殺防守', '放小球': '網前球', '勾球': '網前球', '推球': '推撲球', '撲球': '推撲球'}
        matches['type'] = matches['type'].replace(combined_types)
        print("After converting ball type: ")
        self.print_current_size(matches)

        # Fill zero value in backhand
        matches['backhand'] = matches['backhand'].fillna(value=0)
        matches['backhand'] = matches['backhand'].astype(float).astype(int)

        # Convert ball round type to integer
        matches['ball_round'] = matches['ball_round'].astype(float).astype(int)

        # Standardized area coordinates real court: (355, 960)
        # print(matches['landing_x'].mean(), matches['landing_x'].std())
        # print(matches['landing_y'].mean(), matches['landing_y'].std())
        mean_x, std_x = 175., 82.
        mean_y, std_y = 467., 192.
        matches['landing_x'] = (matches['landing_x']-mean_x) / std_x
        matches['landing_y'] = (matches['landing_y']-mean_y) / std_y
        # print(matches['landing_x'].mean(), matches['landing_x'].std())
        # print(matches['landing_y'].mean(), matches['landing_y'].std())

        self.matches = matches

        return matches

    def drop_na_rally(self, df, columns=[]):
        """Drop rallies which contain na value in columns."""
        df = df.copy()
        for column in columns:
            rallies = df[df[column].isna()]['rally_id']
            df = df[~df['rally_id'].isin(rallies)]
        df = df.reset_index(drop=True)
        return df

    def print_current_size(self, all_match):
        print('\tUnique rally: {}\t Total rows: {}'.format(all_match['rally_id'].nunique(), len(all_match)))


def prepare_dataset(config):
    # directory = './data/set/'
    # filename = 'match.csv'
    # matches = BadmintonCleaner(directory, filename)

    config['filename'] = './data/dataset.csv'
    matches = pd.read_csv(config['filename'])

    # encode shot type
    codes_type, uniques_type = pd.factorize(matches['type'])
    matches['type'] = codes_type + 1                                # Reserve code 0 for paddings
    config['uniques_type'] = uniques_type.to_list()
    config['shot_num'] = len(uniques_type) + 1                      # Add padding

    # encode player
    codes_player, uniques_player = pd.factorize(matches['player'])
    matches['player'] = codes_player + 1                            # Reserve code 0 for paddings
    config['uniques_player'] = uniques_player.to_list()
    config['player_num'] = len(uniques_player) + 1                  # Add padding

    config['folder_name'] = './model/'

    total_train, total_val, total_test = [], [], []

    # use first 80% rallies in the match as train set, others as test set
    group = matches[['rally_id', 'ball_round', 'type', 'landing_x', 'landing_y', 'player', 'set']].groupby('rally_id').apply(lambda r: (r['ball_round'].values, r['type'].values, r['landing_x'].values, r['landing_y'].values, r['player'].values, r['set'].values))

    match_train_indexes, match_test_indexes = [], []
    for match_id in matches['match_id'].unique():
        match = matches[matches['match_id']==match_id]
        rallies = match['rally_id'].unique()
        threshold = int(len(rallies) * 0.8)
        match_train_indexes.append(rallies[:threshold])
        match_test_indexes.append(rallies[threshold:])

    def flatten(t):
        return [item for sublist in t for item in sublist]
    
    train_indexes, test_indexes = flatten(match_train_indexes), flatten(match_test_indexes)

    train_group = group[group.index.isin(train_indexes)]
    test_group = group[group.index.isin(test_indexes)]

    train_dataset = BadmintonDataset(train_group, config['encode_length'], max_ball_round=config['max_ball_round'])
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)

    test_dataset = BadmintonDataset(test_group, config['encode_length'], max_ball_round=config['max_ball_round'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8)
    
    print("Original: ", len(train_group), len(test_group))
    print("Filtering:", len(train_dataset), len(test_dataset))

    total_train.append(train_dataloader), total_test.append(test_dataloader)

    # # k-fold
    # total_train, total_val, total_test = [], [], []
    # kf = KFold(n_splits=config['K'], shuffle=True, random_state=config['seed_value'])
    # for train_fold_indexes, test_indexes in kf.split(train_indexes):
    #     train_group = group[group.index.isin(train_fold_indexes)]
    #     test_group = group[group.index.isin(test_indexes)]

    #     # print("Original: ", len(train_group), len(test_group))

    #     train_dataset = BadmintonDataset(train_group, config['encode_length'], max_ball_round=config['max_ball_round'])
    #     train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)

    #     test_dataset = BadmintonDataset(test_group, config['encode_length'], max_ball_round=config['max_ball_round'])
    #     test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8)
    
    #     # print("Filtering:", len(train_dataset), len(test_dataset))

    #     total_train.append(train_dataloader)
    #     total_test.append(test_dataloader)

    return matches, total_train, total_val, total_test, config