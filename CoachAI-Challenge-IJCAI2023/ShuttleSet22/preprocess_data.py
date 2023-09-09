import os
import re
import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt


VAL_ID = [38, 46, 47, 48, 49]
TEST_ID = [39, 40, 50, 52, 53, 54, 55, 56, 57]


class PreDataProcessor:
    def __init__(self, path: str):
        self.match = pd.read_csv(f"{path}match.csv")
        # convert players to categorical values (anonymize)
        self.show_unique_players()
        1/0
        self.match['winner'] = self.match['winner'].apply(lambda x: self.unique_players.index(x))
        self.match['loser'] = self.match['loser'].apply(lambda x: self.unique_players.index(x))

        self.homography = pd.read_csv(f"{path}homography.csv")
        self.homography = self.homography.drop(columns=['video', 'db'])
        self.homography['set'] = self.match['set']
        self.homography['duration'] = self.match['duration']
        self.homography['winner'] = self.match['winner']
        self.homography['loser'] = self.match['loser']
        self.homography.to_csv(f"{path}match_metadata.csv", index=False)
        self.homography_matrix = pd.read_csv(f"{path}match_metadata.csv", converters={'homography_matrix':lambda x: np.array(ast.literal_eval(x))})

        all_matches = self.read_metadata(directory=f"{path}set")
        cleaned_matches = self.engineer_match(all_matches)
        cleaned_matches.to_csv(f"{path}shot_metadata.csv", index=False)

    def read_metadata(self, directory):
        all_matches = []
        for idx in range(len(self.match)):
            match_idx = self.match['id'][idx]
            match_name = self.match['video'][idx]
            winner = self.match['winner'][idx]
            loser = self.match['loser'][idx]
            current_homography = self.homography_matrix[self.homography_matrix['id'] == match_idx]['homography_matrix'].to_numpy()[0]

            match_path = os.path.join(directory, match_name)
            csv_paths = [os.path.join(match_path, f) for f in os.listdir(match_path) if f.endswith('.csv')]

            one_match = []
            for csv_path in csv_paths:
                data = pd.read_csv(csv_path)
                data['player'] = data['player'].replace(['A', 'B'], [winner, loser])
                data['getpoint_player'] = data['getpoint_player'].replace(['A', 'B'], [winner, winner])
                data['set'] = re.findall(r'\d+', os.path.basename(csv_path))[0]
                one_match.append(data)

            match = pd.concat(one_match, ignore_index=True, sort=False).assign(match_id=match_idx)

            # project screen coordinate to real coordinate
            for i in range(len(match)):
                # project ball coordinates
                p = np.array([match['landing_x'][i], match['landing_y'][i], 1])
                p_real = current_homography.dot(p)
                p_real /= p_real[2]
                match['landing_x'][i], match['landing_y'][i] = round(p_real[0], 1), round(p_real[1], 2)

                # project player coordinates
                p = np.array([match['player_location_x'][i], match['player_location_y'][i], 1])
                p_real = current_homography.dot(p)
                p_real /= p_real[2]
                match['player_location_x'][i], match['player_location_y'][i] = round(p_real[0], 1), round(p_real[1], 2)

                # project opponent coordinates
                p = np.array([match['opponent_location_x'][i], match['opponent_location_y'][i], 1])
                p_real = current_homography.dot(p)
                p_real /= p_real[2]
                match['opponent_location_x'][i], match['opponent_location_y'][i] = round(p_real[0], 1), round(p_real[1], 2)

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

        # Convert landing_area, player location, and opponent location outside to 10 and to integer
        matches = self.drop_na_rally(matches, columns=['landing_area'])
        matches = self.drop_na_rally(matches, columns=['opponent_location_area'])
        for area in outside_area:
            matches.loc[matches['landing_area'] == area, 'landing_area'] = 10
            matches.loc[matches['player_location_area'] == area, 'player_location_area'] = 10
            matches.loc[matches['opponent_location_area'] == area, 'opponent_location_area'] = 10
        matches['landing_area'] = matches['landing_area'].astype(float).astype(int)
        matches['player_location_area'] = matches['player_location_area'].astype(float).astype(int)
        matches['opponent_location_area'] = matches['opponent_location_area'].astype(float).astype(int)
        print("After converting landing_area: ")
        self.print_current_size(matches)

        # Deal with ball type. Convert ball types to general version (10 types)
        # Convert 小平球 to 平球 because of old version
        matches['type'] = matches['type'].replace('小平球', '平球')
        combined_types = {'切球': '切球', '過度切球': '切球', '點扣': '殺球', '殺球': '殺球', '平球': '平球', '後場抽平球': '平球', '擋小球': '接殺防守',
                    '防守回挑': '接殺防守', '防守回抽': '接殺防守', '放小球': '網前球', '勾球': '網前球', '推球': '推撲球', '撲球': '推撲球'}
        shot_type_transform = {
            '發短球': 'short service',
            '長球': 'clear',
            '推撲球': 'push/rush',
            '殺球': 'smash',
            '接殺防守': 'defensive shot',
            '平球': 'drive',
            '網前球': 'net shot',
            '挑球': 'lob',
            '切球': 'drop',
            '發長球': 'long service',
        }
        matches['type'] = matches['type'].replace(combined_types)
        matches['type'] = matches['type'].replace(shot_type_transform)
        print("After converting ball type: ")
        self.print_current_size(matches)

        # Fill zero value in backhand
        matches['backhand'] = matches['backhand'].fillna(value=0)
        matches['backhand'] = matches['backhand'].astype(float).astype(int)

        # Fill zero value in aroundhead
        matches['aroundhead'] = matches['aroundhead'].fillna(value=0)

        # Convert ball round type to integer
        matches['ball_round'] = matches['ball_round'].astype(float).astype(int)

        # Translate lose reasons from Chinese to English (foul is treated as not pass over the net)
        reason_transform = {'出界': 'out', '落點判斷失誤': 'misjudged', '掛網': 'touched the net', '未過網': 'not pass over the net', '對手落地致勝': "opponent's ball landed", '犯規': 'not pass over the net'}
        
        matches['lose_reason'] = matches['lose_reason'].replace(reason_transform)

        mean_x, std_x = 175., 82.
        mean_y, std_y = 467., 192.
        matches['landing_x'] = (matches['landing_x']-mean_x) / std_x
        matches['landing_y'] = (matches['landing_y']-mean_y) / std_y

        # Remove some unrelated fields
        matches = matches.drop(columns=['hit_height', 'hit_area', 'hit_x', 'hit_y', 'win_reason', 'flaw', 'db'])

        return matches

    def compute_statistics(self):
        self.show_unique_players()
        self.compute_matchup_counts()

    def show_unique_players(self):
        # show players
        column_players = self.match[['winner', 'loser']].values.ravel()
        self.unique_players = pd.unique(column_players).tolist()
        print(self.unique_players, len(self.unique_players))

    def compute_matchup_counts(self):
        # compute matchup counts of each player
        column_values = self.match[['winner', 'loser']].values
        players = []
        for column_value in column_values:
            players.append(column_value)

        player_matrix = [[0] * len(self.unique_players) for _ in range(len(self.unique_players))]
        for player in players:
            player_index_row, player_index_col = self.unique_players.index(player[0]), self.unique_players.index(player[1])
            player_matrix[player_index_row][player_index_col] += 1
            player_matrix[player_index_col][player_index_row] += 1
        player_matrix = pd.DataFrame(player_matrix, index=self.unique_players, columns=self.unique_players)
        
        plot = sns.heatmap(player_matrix, annot=True, linewidths=0.5, cbar=False)
        plt.xticks(rotation=30, ha='right')
        plot.get_figure().savefig("../figures/player_matrix.png", dpi=300, bbox_inches='tight')
        plot.clear()

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


class CoachAITrainTestSplit:
    def __init__(self, path):
        self.metadata = pd.read_csv(f"{path}shot_metadata.csv")
        self.matches = pd.read_csv(f"{path}match_metadata.csv")
        self.given_strokes_num = 4

        match_train, match_val, match_test = [], [], []
        for match_id in self.metadata['match_id'].unique():
            if match_id in VAL_ID:
                match_val.append(self.metadata[self.metadata['match_id']==match_id])
            elif match_id in TEST_ID:
                match_test.append(self.metadata[self.metadata['match_id']==match_id])
            else:
                match_train.append(self.metadata[self.metadata['match_id']==match_id])

        match_train = pd.concat(match_train, ignore_index=True, sort=False)
        match_val = pd.concat(match_val, ignore_index=True, sort=False)
        match_test = pd.concat(match_test, ignore_index=True, sort=False)
        print(match_train['rally_id'].nunique(), match_val['rally_id'].nunique(), match_test['rally_id'].nunique())

        match_train = self.preprocess_files(match_train)
        match_test = self.preprocess_files(match_test)
        match_val = self.preprocess_files(match_val)

        print("========== Val not in Train=========")
        for player in match_val['player'].unique():
            if player not in match_train['player'].unique():
                print(player, sep=', ')
        print("========== Test not in Train=========")
        for player in match_test['player'].unique():
            if player not in match_train['player'].unique():
                print(player, sep=', ')

        # output to csv
        match_train.to_csv(f"{path}train.csv", index=False)
        match_test[match_test['ball_round'].isin([_ for _ in range(self.given_strokes_num+1)])].to_csv(f"{path}test_given.csv", index=False)
        match_val[match_val['ball_round'].isin([_ for _ in range(self.given_strokes_num+1)])].to_csv(f"{path}val_given.csv", index=False)
        match_test[~match_test['ball_round'].isin([_ for _ in range(self.given_strokes_num+1)])][['rally_id', 'ball_round', 'type', 'landing_x', 'landing_y']].to_csv(f"{path}test_gt.csv", index=False)
        match_val[~match_val['ball_round'].isin([_ for _ in range(self.given_strokes_num+1)])][['rally_id', 'ball_round', 'type', 'landing_x', 'landing_y']].to_csv(f"{path}val_gt.csv", index=False)

    def preprocess_files(self, match):
        def flatten(t):
            return [item for sublist in t for item in sublist]

        unused_columns = ['server']
        # , 'player_location_area', 'player_location_x', 'player_location_y', 'opponent_location_area', 'opponent_location_x', 'opponent_location_y'

        # compute rally length
        rally_len = []
        for rally_id in match['rally_id'].unique():
            rally_info = match.loc[match['rally_id'] == rally_id]
            rally_len.append([len(rally_info)]*len(rally_info))
        rally_len = flatten(rally_len)
        match['rally_length'] = rally_len

        # filter rallies that are less than \tau + 1
        match = match[match['rally_length'] >= self.given_strokes_num+1].reset_index(drop=True)

        return match.drop(columns=unused_columns)


if __name__ == "__main__":
    path = "../data/"
    data_processor = PreDataProcessor(path=path)
    data_processor.compute_statistics()

    data_splitter = CoachAITrainTestSplit(path=path)