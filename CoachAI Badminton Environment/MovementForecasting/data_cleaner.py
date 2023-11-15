import pandas as pd
import numpy as np
import ast
import os
import re

def drop_na_rally(df, columns=[]):
    """Drop rallies which contain na value in columns."""
    df = df.copy()
    for column in columns:
        rallies = df[df[column].isna()]['rally_id']
        df = df[~df['rally_id'].isin(rallies)]
    df = df.reset_index(drop=True)
    return df

def preprocess_data(matches):
    matches['rally_id'] = matches.groupby(['match_id', 'set', 'rally']).ngroup()

    # Drop flaw rally
    if 'flaw' in matches.columns:
        flaw_rally = matches[matches['flaw'].notna()]['rally_id']
        matches = matches[~matches['rally_id'].isin(flaw_rally)]
        matches = matches.reset_index(drop=True)

    # Drop unknown ball type
    unknown_rally = matches[matches['type'] == '未知球種']['rally_id']
    matches = matches[~matches['rally_id'].isin(unknown_rally)]
    matches = matches.reset_index(drop=True)

    # Drop hit_area at outside
    outside_area = [10, 11, 12, 13, 14, 15, 16]
    matches.loc[matches['server'] == 1, 'hit_area'] = 7
    for area in outside_area:
        outside_rallies = matches.loc[matches['hit_area'] == area, 'rally_id']
        matches = matches[~matches['rally_id'].isin(outside_rallies)]
        matches = matches.reset_index(drop=True)
    # Deal with hit_area convert hit_area to integer
    matches = drop_na_rally(matches, columns=['hit_area'])
    matches['hit_area'] = matches['hit_area'].astype(float).astype(int)

    # Convert landing_area outside to 10 and to integer
    matches = drop_na_rally(matches, columns=['landing_area'])
    for area in outside_area:
        matches.loc[matches['landing_area'] == area, 'landing_area'] = 10
    matches['landing_area'] = matches['landing_area'].astype(float).astype(int)

    # Deal with ball type. Convert ball types to general version (10 types)
    # Convert 小平球 to 平球 because of old version
    matches['type'] = matches['type'].replace('小平球', '平球')
    combined_types = {'切球': '切球', '過度切球': '切球', '點扣': '殺球', '殺球': '殺球', '平球': '平球', '後場抽平球': '平球', '擋小球': '接殺防守',
                '防守回挑': '接殺防守', '防守回抽': '接殺防守', '放小球': '網前球', '勾球': '網前球', '推球': '推撲球', '撲球': '推撲球'}
    matches['type'] = matches['type'].replace(combined_types)

    # Fill zero value in backhand
    matches['backhand'] = matches['backhand'].fillna(value=0)
    matches['backhand'] = matches['backhand'].astype(float).astype(int)

    # Fill zero value in aroundhead
    matches['aroundhead'] = matches['aroundhead'].fillna(value=0)
    matches['aroundhead'] = matches['aroundhead'].astype(float).astype(int)

    # Convert ball round type to integer
    matches['ball_round'] = matches['ball_round'].astype(float).astype(int)

    # Standardized area coordinates real court: (355, 960)
    mean_x, std_x = 175., 82.
    mean_y, std_y = 467., 192.
    matches['player_location_x'] = (matches['player_location_x'] - mean_x) / std_x
    matches['player_location_y'] = (matches['player_location_y'] - mean_y) / std_y

    matches['opponent_location_x'] = (matches['opponent_location_x'] - mean_x) / std_x
    matches['opponent_location_y'] = (matches['opponent_location_y'] - mean_y) / std_y
    
    return matches

def DataCleaner(args):
    if args['already_have_data']:
        return pd.read_csv(args['preprocessed_data_path'])
    
    data_folder = args['input_data_folder_path']
    match_list_csv = args['match_list_csv']
    homography_matrix_list_csv = args['homography_matrix_list_csv']
    output_path = args['prepared_data_output_path']

    match_list = pd.read_csv(data_folder + match_list_csv)

    homography_matrix_list = pd.read_csv(data_folder + homography_matrix_list_csv, converters={'homography_matrix':lambda x: np.array(ast.literal_eval(x))})
    
    available_matches = []
    for idx in range(len(match_list)):
        match_idx = match_list['id'][idx]
        match_name = match_list['video'][idx]
        winner = match_list['winner'][idx]
        loser = match_list['loser'][idx]

        homography_matrix = homography_matrix_list[homography_matrix_list['id'] == match_idx]['homography_matrix'].to_numpy()[0]

        match_folder = os.path.join(data_folder, match_name)
        set_csv = [os.path.join(match_folder, f) for f in os.listdir(match_folder) if f.endswith('.csv')]
        
        match_data = []
        for csv in set_csv:
            set_data = pd.read_csv(csv)
            set_data['player'] = set_data['player'].replace(['A', 'B'], [winner, loser])
            set_data['set'] = re.findall(r'\d+', os.path.basename(csv))[0]
            match_data.append(set_data)

        match = pd.concat(match_data, ignore_index=True, sort=False).assign(match_id=match_idx)
        
        # project screen coordinate to real coordinate
        for i in range(len(match)):
            player_location = np.array([match['player_location_x'][i], match['player_location_y'][i], 1])
            player_location_real = homography_matrix.dot(player_location)
            player_location_real /= player_location_real[2]

            match.iloc[i, match.columns.get_loc('player_location_x')] = player_location_real[0]
            match.iloc[i, match.columns.get_loc('player_location_y')] = player_location_real[1]
            
            opponent_location = np.array([match['opponent_location_x'][i], match['opponent_location_y'][i], 1])
            opponent_location_real = homography_matrix.dot(opponent_location)
            opponent_location_real /= opponent_location_real[2]

            match.iloc[i, match.columns.get_loc('opponent_location_x')] = opponent_location_real[0]
            match.iloc[i, match.columns.get_loc('opponent_location_y')] = opponent_location_real[1]

            landing_location = np.array([match['landing_x'][i], match['landing_y'][i], 1])
            landing_location_real = homography_matrix.dot(landing_location)
            landing_location_real /= landing_location_real[2]

            match.iloc[i, match.columns.get_loc('landing_x')] = landing_location_real[0]
            match.iloc[i, match.columns.get_loc('landing_y')] = landing_location_real[1]


        available_matches.append(match)
    
    available_matches = pd.concat(available_matches, ignore_index=True, sort=False)
    
    cleaned_matches = preprocess_data(available_matches)
    cleaned_matches.to_csv(args['prepared_data_output_path'], index=False)

    return cleaned_matches