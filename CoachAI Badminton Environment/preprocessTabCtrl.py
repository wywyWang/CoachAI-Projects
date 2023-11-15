# This Python file uses the following encoding: utf-8
import os
import re

from PySide6.QtWidgets import QLineEdit, QFileDialog
from PySide6.QtCore import Qt, Signal, Slot, QThread
import pandas as pd
import ast
import numpy as np

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from GUI.PreprocessWidget import PreprocessWidget


class PreprocessWidgetCtrl:
    def __init__(self, widget: PreprocessWidget):
        self.parent = widget
        self.ui = widget.ui

        # file input
        self.ui.load_directory.clicked.connect(self.setDirectory)
        self.ui.load_homography.clicked.connect(lambda: self.setFile(self.ui.homography_filename))
        self.ui.load_list.clicked.connect(lambda: self.setFile(self.ui.list_filename))

        self.ui.confirm.clicked.connect(self.preprocessData)

    # event function for selecting folder
    @Slot()
    def setDirectory(self):
        foldername = QFileDialog.getExistingDirectory(self.parent, caption="Open folder", dir="./")
        if len(foldername) == 0:
            return
        
        self.ui.directory.setText(foldername)

    # event function for selecting file
    @Slot(QLineEdit)
    def setFile(self, target: QLineEdit):
        directory = self.ui.directory.text()
        filename, _ = QFileDialog.getOpenFileName(self.parent, 
                                                  caption="Open file", 
                                                  dir= directory if os.path.exists(directory) else "./", 
                                                  filter="csv file(*.csv)")
        if len(filename) == 0:
            return
        
        target.setText(filename)

    @Slot()
    def preprocessData(self):
        if not os.path.exists(self.ui.directory.text()):
            self.ui.error_message.setText('directory not exist')
            return
        if not os.path.exists(self.ui.homography_filename.text()):
            self.ui.error_message.setText('homography file not exit')
            return
        if not os.path.exists(self.ui.list_filename.text()):
            self.ui.error_message.setText('index file not exist')
            return
        
        if self.ui.output_filename.text() == "":
            self.ui.error_message.setText("output filename can't be empty")
            return
        
        self.ui.confirm.setEnabled(False)
        self.preprocess_thread = PreprocessThread(self.ui.directory.text(), 
                                                  self.ui.list_filename.text(), 
                                                  self.ui.homography_filename.text(),
                                                  self.ui.output_filename.text())
        self.updateLoadingProgress(0)
        self.preprocess_thread.loadingInfoUpdate.connect(self.updateLoadingInfo)
        self.preprocess_thread.loadingProgressUpdate.connect(self.updateLoadingProgress)
        self.preprocess_thread.finished.connect(self.threadFinished)
        self.preprocess_thread.start()

    # event function for preprocess thread to update current action text
    @Slot(str)
    def updateLoadingInfo(self, info: str):
        self.ui.progress_message.setText(info)

    # event function for preprocess thread to update current action progress
    @Slot(float)
    def updateLoadingProgress(self, value: float):
        self.ui.progressBar.setValue(int(value))

    # event function for preprocess thread to allow next input since current is finish
    @Slot()
    def threadFinished(self):
        self.ui.confirm.setEnabled(True)
        
class PreprocessThread(QThread):
    loadingInfoUpdate = Signal(str)
    loadingProgressUpdate = Signal(float)
    finished = Signal()

    def __init__(self, directory, list, homographary, output, parent=None):
        super().__init__(parent)
        self.directory = directory
        self.list_filepath = list
        self.homography_filepath = homographary
        self.output_filepath = output

        pd.options.mode.chained_assignment = None


    def run(self):
        self.loadingInfoUpdate.emit('initializing...')
        available_matches = pd.read_csv(self.list_filepath)
        self.homography_matrix = pd.read_csv(self.homography_filepath, converters={
                                             'homography_matrix': lambda x: np.array(ast.literal_eval(x))})
        
        self.loadingInfoUpdate.emit(f'{len(available_matches)} matches found.')
        all_matches = self.read_match(self.directory, available_matches)

        self.loadingInfoUpdate.emit(f'process data...')
        self.loadingProgressUpdate.emit(97)
        cleaned_matches = self.engineer_match(all_matches)
        cleaned_matches.to_csv(self.output_filepath, index=False)
        self.loadingProgressUpdate.emit(100)
        self.loadingInfoUpdate.emit(f'output to {self.output_filepath}')

        self.finished.emit()

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
            self.loadingInfoUpdate.emit(f'read matches {idx}/{len(available_matches)}')
            self.loadingProgressUpdate.emit(99*idx/len(available_matches))
            match_idx = available_matches['id'][idx]
            match_name = available_matches['video'][idx]
            winner = available_matches['winner'][idx]
            loser = available_matches['loser'][idx]
            current_homography = self.homography_matrix[self.homography_matrix['id'] == match_idx]['homography_matrix'].to_numpy()[
                0]

            match_path = os.path.join(directory, match_name)
            csv_paths = [os.path.join(match_path, f) for f in os.listdir(
                match_path) if f.endswith('.csv')]

            one_match = []
            for csv_path in csv_paths:
                data = pd.read_csv(csv_path)
                data['player'] = data['player'].replace(
                    ['A', 'B'], [winner, loser])
                data['set'] = re.findall(r'\d+', os.path.basename(csv_path))[0]
                one_match.append(data)

            match = pd.concat(one_match, ignore_index=True,
                              sort=False).assign(match_id=match_idx)
            
            def homography(p):
                p_real = current_homography.dot(p)
                p_real /= p_real[2]
                return p_real[0], p_real[1]
            
            # project screen coordinate to real coordinate
            for i in range(len(match)):
                match['landing_x'][i], match['landing_y'][i] = homography(np.array([match['landing_x'][i], match['landing_y'][i], 1]))
                match['hit_x'][i], match['hit_y'][i] = homography(np.array([match['hit_x'][i], match['hit_y'][i], 1]))
                match['player_location_x'][i], match['player_location_y'][i] = homography(np.array([match['player_location_x'][i], match['player_location_y'][i], 1]))
                match['opponent_location_x'][i], match['opponent_location_y'][i] = homography(np.array([match['opponent_location_x'][i], match['opponent_location_y'][i], 1]))

            all_matches.append(match)

        all_matches = pd.concat(all_matches, ignore_index=True, sort=False)
        return all_matches
    
    def engineer_match(self, matches):
        matches['rally_id'] = matches.groupby(
            ['match_id', 'set', 'rally']).ngroup()
        print("Original: ")
        self.print_current_size(matches)

        self.loadingInfoUpdate.emit(f'Drop flaw rally')
        # Drop flaw rally
        if 'flaw' in matches.columns:
            flaw_rally = matches[matches['flaw'].notna()]['rally_id']
            matches = matches[~matches['rally_id'].isin(flaw_rally)]
            matches = matches.reset_index(drop=True)
        print("After Dropping flaw: ")
        self.print_current_size(matches)

        self.loadingInfoUpdate.emit(f'Drop unknown ball type')
        # Drop unknown ball type
        unknown_rally = matches[matches['type'] == '未知球種']['rally_id']
        matches = matches[~matches['rally_id'].isin(unknown_rally)]
        matches = matches.reset_index(drop=True)
        print("After dropping unknown ball type: ")
        self.print_current_size(matches)

        self.loadingInfoUpdate.emit(f'Drop hit_area at outside')
        # Drop hit_area at outside
        outside_area = [10, 11, 12, 13, 14, 15, 16]
        matches.loc[matches['server'] == 1, 'hit_area'] = 7
        for area in outside_area:
            outside_rallies = matches.loc[matches['hit_area']
                                          == area, 'rally_id']
            matches = matches[~matches['rally_id'].isin(outside_rallies)]
            matches = matches.reset_index(drop=True)

        self.loadingInfoUpdate.emit(f'Deal with hit_area convert hit_area to integer')
        # Deal with hit_area convert hit_area to integer
        matches = self.drop_na_rally(matches, columns=['hit_area'])
        matches['hit_area'] = matches['hit_area'].astype(float).astype(int)
        print("After converting hit_area: ")
        self.print_current_size(matches)

        self.loadingInfoUpdate.emit(f'Convert landing_area outside to 10 and to integer')
        # Convert landing_area outside to 10 and to integer
        matches = self.drop_na_rally(matches, columns=['landing_area'])
        for area in outside_area:
            matches.loc[matches['landing_area'] == area, 'landing_area'] = 10
        matches['landing_area'] = matches['landing_area'].astype(
            float).astype(int)
        print("After converting landing_area: ")
        self.print_current_size(matches)

        self.loadingInfoUpdate.emit(f'Deal with ball type convertion')
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

        #self.loadingInfoUpdate.emit(f'Standardized coord')
        #self.loadingProgressUpdate.emit(98)
        # Standardized area coordinates real court: (355, 960)
        # print(matches['landing_x'].mean(), matches['landing_x'].std())
        # print(matches['landing_y'].mean(), matches['landing_y'].std())
        #mean_x, std_x = 175., 82.
        #mean_y, std_y = 467., 192.
        #matches['landing_x'] = (matches['landing_x']-mean_x) / std_x
        #matches['landing_y'] = (matches['landing_y']-mean_y) / std_y
        #matches['hit_x'] = (matches['hit_x']-mean_x) / std_x
        #matches['hit_y'] = (matches['hit_y']-mean_y) / std_y
        #self.loadingProgressUpdate.emit(99)
        #matches['player_location_x'] = (matches['player_location_x']-mean_x) / std_x
        #matches['player_location_y'] = (matches['player_location_y']-mean_y) / std_y
        #matches['opponent_location_x'] = (matches['opponent_location_x']-mean_x) / std_x
        #matches['opponent_location_y'] = (matches['opponent_location_y']-mean_y) / std_y
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
        print('\tUnique rally: {}\t Total rows: {}'.format(
            all_match['rally_id'].nunique(), len(all_match)))

                