# This Python file uses the following encoding: utf-8
import os
import re

from PySide6.QtWidgets import QWidget, QLineEdit, QFileDialog, QTableView, QAbstractItemView, QHeaderView, QComboBox
from PySide6.QtCore import Qt, Signal, Slot, QThread
from PySide6.QtGui import QIntValidator
from PySide6 import QtGui 
import pandas as pd
import ast
import numpy as np
import pickle
from copy import deepcopy
from typing import List, Tuple, Sequence, Literal

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from GUI.GenerateWidget import GenerateWidget

class GenerateWidgetCtrl:
    def __init__(self, widget: GenerateWidget):
        self.parent = widget
        self.ui = widget.ui

        # grouped up path and path chooser, so that we can enable/disable them conveniently
        self.custom_model1 = [self.ui.model1_custom_path, self.ui.model1_load_custom]
        self.custom_model2 = [self.ui.model2_custom_path, self.ui.model2_load_custom]

        # rally count can only be positive int
        only_int = QIntValidator()
        only_int.setBottom(1)
        self.ui.rally_count.setValidator(only_int)

        # enable/disable custom model chooser when the chosen item change
        self.ui.model1_custom.toggled.connect(lambda: self.modelChange(self.ui.model1_custom.isChecked(), self.custom_model1))
        self.ui.model2_custom.toggled.connect(lambda: self.modelChange(self.ui.model2_custom.isChecked(), self.custom_model2))
        self.ui.model1_ShuttleNet.toggled.connect(lambda: self.modelChange(self.ui.model1_ShuttleNet.isChecked(), [self.ui.model1_ShuttleNet_player]))
        self.ui.model2_ShuttleNet.toggled.connect(lambda: self.modelChange(self.ui.model2_ShuttleNet.isChecked(), [self.ui.model2_ShuttleNet_player]))

        # file input
        self.ui.model1_load_custom.clicked.connect(lambda: self.setFile(self.ui.model1_custom_path,"pickle file(*.pkl)"))
        self.ui.model2_load_custom.clicked.connect(lambda: self.setFile(self.ui.model2_custom_path,"pickle file(*.pkl)"))

        self.ui.confirm.clicked.connect(self.generateData)

        # possible opponenets id for shuttleNet to choose
        self.opponents = {'Kento MOMOTA': 0, 'CHOU Tien Chen': 1, 'Anthony Sinisuka GINTING': 2, 'CHEN Long': 3, 'CHEN Yufei': 4, 
                     'TAI Tzu Ying': 5, 'Viktor AXELSEN': 6, 'Anders ANTONSEN': 7, 'PUSARLA V. Sindhu': 8, 'WANG Tzu Wei': 9, 
                     'Khosit PHETPRADAB': 10, 'Jonatan CHRISTIE': 11, 'NG Ka Long Angus': 12, 'SHI Yuqi': 13, 'Ratchanok INTANON': 14, 
                     'An Se Young': 15, 'Busanan ONGBAMRUNGPHAN': 16, 'Mia BLICHFELDT': 17, 'LEE Zii Jia': 18, 'LEE Cheuk Yiu': 19, 
                     'Rasmus GEMKE': 20, 'Michelle LI': 21, 'Supanida KATETHONG': 22, 'Carolina MARIN': 23, 'Pornpawee CHOCHUWONG': 24, 
                     'Sameer VERMA': 25, 'Neslihan YIGIT': 26, 'Hans-Kristian Solberg VITTINGHUS': 27, 'LIEW Daren': 28, 
                     'Evgeniya KOSETSKAYA': 29, 'KIDAMBI Srikanth': 30, 'Soniia CHEAH': 31, 'Gregoria Mariska TUNJUNG': 32, 
                     'Akane YAMAGUCHI': 33, 'HE Bingjiao': 34, 
                     #'胡佑齊': 35, '張允澤': 36, '許喆宇': 37, '陳政佑': 38, '林祐賢': 39, '李佳豪': 40, 
                     'LOH Kean Yew': 41, 'Lakshya SEN': 42, 'Kunlavut VITIDSARN': 43, 'WANG Hong Yang': 44, 
                     'Kodai NARAOKA': 45, 'JEON Hyeok Jin': 46, 'Wen Chi Hsu': 47, 'Nozomi Okuhara': 48, 'WANG Zhi Yi': 49, 
                     'PRANNOY H. S.': 50, 'Chico Aura DWI WARDOYO': 51, 'LU Guang Zu': 52, 'ZHAO Jun Peng': 53, 'Kenta NISHIMOTO': 54, 
                     'NG Tze Yong': 55, 'Victor SVENDSEN': 56, 'WENG Hong Yang': 57, 'Aakarshi KASHYAP': 58, 'LI Shi Feng': 59, 
                     'KIM Ga Eun': 60, 'HAN Yue': 61, 'Other': 62, 'NYCU': 63}
        self.ui.model1_ShuttleNet_player.addItems(self.opponents.keys())
        self.ui.model2_ShuttleNet_player.addItems(self.opponents.keys())

        # display drop down menu contained id as well se opponent to choose
        model = QtGui.QStandardItemModel(0, 2)
        for name, id in self.opponents.items():
            model.appendRow([QtGui.QStandardItem(f"{id}"), 
                             QtGui.QStandardItem(name)])
        self.setupOpponentChooser(self.ui.model1_ShuttleNet_player, model)
        self.setupOpponentChooser(self.ui.model2_ShuttleNet_player, model)

    def setupOpponentChooser(self, menu:QComboBox, model: QtGui.QStandardItemModel):
        menu.setModel(model)
        menu.setModelColumn(1)
        view = QTableView(menu)
        menu.setView(view)
        
        view.setColumnWidth(0, 20)
        view.setColumnWidth(1, 230)
        view.setMinimumWidth(247)
        view.verticalHeader().hide()
        view.horizontalHeader().hide()
        header = view.horizontalHeader()
        view.setStyleSheet("QTableView::item { border: none; }")
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        for row in range(view.model().rowCount()):
            view.setRowHeight(row, 8)

    """
    Slot function for file chooser, given filter can target, 
    it will set the return file path to target
    """
    @Slot(QLineEdit, str)
    def setFile(self, target: QLineEdit, filter:str):
        filename, _ = QFileDialog.getOpenFileName(self.parent, 
                                                  caption="Open file", 
                                                  dir= "D:/CoachAI/demo/", 
                                                  filter=filter)
        if len(filename) == 0:
            return
        
        target.setText(filename)

    """
    Slot function executed when started to generate data
    """
    @Slot()
    def generateData(self):
        if self.ui.model1_custom.isChecked() and not os.path.exists(self.ui.model1_custom_path.text()):
            self.ui.error_message.setText('model1 path not exist')
            return
        
        if self.ui.model2_custom.isChecked() and not os.path.exists(self.ui.model2_custom_path.text()):
            self.ui.error_message.setText('model2 path not exist')
            return
        
        if self.ui.output_filename.text() == "":
            self.ui.error_message.setText("output filename can't be empty")
            return
        
        self.ui.confirm.setEnabled(False)
        model1_shuttleNet_player = self.opponents[self.ui.model1_ShuttleNet_player.currentText()]
        model2_shuttleNet_player = self.opponents[self.ui.model2_ShuttleNet_player.currentText()]
        self.generate_thread = GenerateThread2Agent(int(self.ui.rally_count.text()),
                                                    self.ui.model1_custom_path.text(), self.ui.model1_ShuttleNet.isChecked(), model1_shuttleNet_player,
                                                    self.ui.model2_custom_path.text(), self.ui.model2_ShuttleNet.isChecked(), model2_shuttleNet_player,
                                                    self.ui.output_filename.text())
        
        self.ui.progressBar.setValue(0)
        self.generate_thread.progressUpdate.connect(self.updateProgressBar)
        self.generate_thread.finished.connect(self.threadFinished)
        self.generate_thread.start()

    @Slot(int, int)
    def updateProgressBar(self, currentValue:int, maxValue:int):
        self.ui.progressBar.setValue(int(currentValue/maxValue*100))

    @Slot(bool, list)
    def modelChange(self, checked:bool, widget_list: List[QWidget]):
        for widget in widget_list:
            widget.setEnabled(checked)


    # event function for preprocess thread to allow next input since current is finish
    @Slot()
    def threadFinished(self):
        self.ui.confirm.setEnabled(True)

class GenerateThread(QThread):
    progressUpdate = Signal(int, int)
    finished = Signal()

    def __init__(self, output_filename:str, parent=None):
        super().__init__(parent)
        self.output_filename = output_filename
        self.shottype_mapping = ['發短球', '長球', '推球', '殺球', '擋小球', '平球', '放小球', '挑球', '切球', '發長球']
        self.round = 1
        self.rally = 0
        self.output = pd.DataFrame()

    def coordContinuous2Discrete(self, x: float, y: float, position: Literal[1, 2]):
        if position == 1:
            pass
        elif position == 2:
            x = -x
            y = -y
        else:
            raise NotImplementedError

        if y < 0:        return 10
        elif y < 110: region = [7,8,9]
        elif y < 220: region = [4,5,6]
        elif y < 330: region = [1,2,3]
        else:            return 10
        
        if x < -127.5:  return 10
        elif x < -42.5: return region[0]
        elif x < 42.5:  return region[1]
        elif x < 127.5: return region[2]
        else:           return 10
    
    # convert action, stage based data to rally base
    def dumpData(self, player:int, state: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]], 
                 action: Tuple[int, Tuple[float, float], Tuple[float, float]], 
                 action_prob:Tuple[list, Tuple[list, list, list], Tuple[list, list, list]], reward: int, is_launch: bool):
        # launch failed
        if is_launch and reward == -1:
            return
        state_player, state_opponent, state_ball = state
        action_type, action_land, action_move = action
        action_type_prob, action_land_gmm_param, action_move_gmm_param = action_prob

        discrete_state = (self.coordContinuous2Discrete(*state_player, 2),
                          self.coordContinuous2Discrete(*state_opponent, 1),
                          self.coordContinuous2Discrete(*state_ball, 2))

        if is_launch:
            self.rally += 1

        # 11 mean cannot reach, prev state is last state
        if action_type != 11 and action_land[1] > 0:
            print(action_type)
            player_x, player_y = action_move
            landing_x, landing_y = action_land
            row = pd.DataFrame([{'rally':self.rally,
                                'obs_ball_round':self.round,
                                'obs_player': 'A' if player == 1 else 'B',
                                'obs_serve': is_launch,
                                'act_landing_location_x':landing_x,
                                'act_landing_location_y':landing_y,
                                'act_player_defend_x': player_x,
                                'act_player_defend_y':player_y,
                                'act_ball_type':self.shottype_mapping[action_type-1],
                                'shot_prob': action_type_prob,
                                'land_prob': action_land_gmm_param,
                                'move_prob': action_move_gmm_param,
                                'state': discrete_state}])
            self.output = pd.concat([self.output, row])
        
        self.round += 1

        # current rally end
        if reward == -1:
            self.round = 1

    def save(self):
        self.output.to_csv(self.output_filename, index=False)

class GenerateThread2Agent(GenerateThread):
    def __init__(self, rally_count:int, model1_path:str, is_model1_shuttleNet: bool, model1_shuttleNet_player: int,
                 model2_path:str, is_model2_shuttleNet: bool, model2_shuttleNet_player: int,
                 output_filename: str, parent=None):
        super().__init__(output_filename, parent)
        self.output_filename = output_filename
        self.model1_path = model1_path
        self.model2_path = model2_path
        self.is_model1_shuttleNet = is_model1_shuttleNet
        self.is_model2_shuttleNet = is_model2_shuttleNet
        self.model1_score = 0
        self.model2_score = 0
        
        self.model1_shuttleNet_player = model1_shuttleNet_player
        self.model2_shuttleNet_player = model2_shuttleNet_player

        self.rally_count = rally_count

        # get first n ball from history data
        # needed ball round + 1 and filtered out last row so that last ball is not end
        if self.is_model1_shuttleNet or self.is_model2_shuttleNet:
            print('contain ShuttleNet')
            self.init_row_count = 2 # for shuttleNet
        else:
            self.init_row_count = 2
        data = pd.read_csv('StrokeForecasting/data/continous_subjective.csv')
        data = data[['rally_id','type','rally','ball_round', 'landing_x', 'landing_y',
                     'player_location_x', 'player_location_y', 'opponent_location_x','opponent_location_y']]
        grouped = data.groupby(['rally_id'])
        filtered = grouped.filter(lambda x: len(x) >= self.init_row_count)
        data.dropna(inplace=True)
        self.history_data = filtered.groupby(['rally_id']).head(self.init_row_count)
        self.group_keys = list(self.history_data.groupby(['rally_id']).groups.keys())

        self.type_mapping = {'發短球': 1, '長球': 2, '推撲球': 3, '殺球':4, '接殺防守':5, '平球':6, '網前球':7, '挑球':8, '切球':9, '發長球':10, '接不到':11} 

    # sample start state from history data
    def sampleStartState(self):
        self.states = []
        self.actions = []
        self.actions_prob = []

        random_group_index = np.random.choice(len(self.group_keys))
        rows = self.history_data.groupby(['rally_id']).get_group(self.group_keys[random_group_index])

        for i, (index, row) in enumerate(rows.iterrows()):
            player_coord = row['player_location_x'], row['player_location_y']
            opponent_coord = row['opponent_location_x'], row['opponent_location_y']
            landing_coord = row['landing_x'], row['landing_y']
            type = self.type_mapping[row['type']]

            if i == 0:
                state = (player_coord, opponent_coord, player_coord)
                self.states.append(state)
            else:
                state = (player_coord, opponent_coord, prev_landing_coord)
                action = (prev_type, prev_landing_coord, 
                          (-row['opponent_location_x'], -row['opponent_location_y']))
                self.states.append(state)
                self.actions.append(action)
                self.actions_prob.append(([],[],[]))


            prev_landing_coord = landing_coord
            prev_opponent_coord = opponent_coord
            prev_type = type

        #filtered last row to avoid data contain end ball
        self.states = self.states[:-1]
        self.actions = self.actions[:-1]
        self.actions_prob = self.actions_prob[:-1]

    def outputScore(self):
        pass

        # tqdm.tqdm.write(f'{self.model1_score}:{self.model2_score}')
        # with open('score.txt', 'a') as file:
        #     file.write(f'{self.model1_score}:{self.model2_score}\n')

    def isGameEnd(self):
        if self.model1_score < 21 and self.model2_score < 21:
            return False
        if self.model1_score == 30 or self.model2_score == 30:
            return True
        if abs(self.model1_score - self.model2_score) < 2:
            return False    
        return True

    def run(self):
        from RLEnvironment import Env
        from SuperviseAgent import SuperviseAgent
        import debugpy
        debugpy.debug_this_thread()
        if self.is_model1_shuttleNet:
            self.model1 = SuperviseAgent(self.model1_shuttleNet_player, 1)
        else:
            with open(self.model1_path, 'r+b') as model:
                self.model1 = pickle.load(model)
        if self.is_model2_shuttleNet:
            self.model2 = SuperviseAgent(self.model2_shuttleNet_player, 2)
        else:
            with open(self.model2_path, 'r+b') as model:
                self.model2 = pickle.load(model)
        self.env = Env()

        self.sampleStartState()
        turn = 1
        print(self.states)
        print(self.actions)
        #while not self.isGameEnd():
        launcher = 1
        is_launch = True
        for i in range(self.rally_count):
            self.progressUpdate.emit(i+1, self.rally_count)
            if turn == 1:
                if self.is_model1_shuttleNet:
                    action, action_prob = self.model1.action(self.states, self.actions)
                else:
                    action, action_prob = self.model1.action(self.states[-1], is_launch)
                #print(action_prob)
                if action[0] == 11:
                    print('cannot reach')
                state, reward = self.env.step(action, is_launch)
                if reward != -1:
                    self.states.append(state)
                if action is not None:
                    self.actions.append(action)
                    self.actions_prob.append(action_prob)
                turn = 2
                is_launch = False

                # round end
                if reward == -1:
                    is_launch = True
                    #print(len(self.states))
                    turn_ = launcher

                    # output data to dataFrame
                    for i, (state, action, action_prob) in enumerate(zip(self.states, self.actions, self.actions_prob)):
                        if i == len(self.states) - 1:
                            self.dumpData(turn_, state, action, action_prob, -1, i == 0) # last rally of round reward is -1
                        elif i == 0:
                            self.dumpData(turn_, state, action, action_prob, 0, True)
                        else:
                            self.dumpData(turn_, state, action, action_prob, 0, False)
                        turn_ = 2 if turn_ == 1 else 1
                    self.sampleStartState()
                    self.env.reset(self.states[-1])
                    launcher = 2
                    
                    self.model2_score += 1
                    # print(f'rally end, next launcher{launcher}')
                    if self.isGameEnd():
                        self.outputScore()
                        self.model1_score = 0
                        self.model2_score = 0
            elif turn == 2:
                if self.is_model2_shuttleNet:
                    action, action_prob = self.model2.action(self.states, self.actions)
                else:
                    action, action_prob = self.model2.action(self.states[-1], is_launch)
                #print(action_prob)
                next_state, reward = self.env.step(action, is_launch)
                if next_state is not None:
                    self.states.append(next_state)
                if action is not None:
                    self.actions.append(action)
                    self.actions_prob.append(action_prob)
                if action[0] == 11:
                    print('cannot reach')
                turn = 1
                is_launch = False

                # round end
                if reward == -1:
                    is_launch = True
                    turn_ = launcher

                    # output data to dataFrame
                    for i, (state, action, action_prob) in enumerate(zip(self.states, self.actions, self.actions_prob)):
                        if i == len(self.states) - 1:
                            self.dumpData(turn_, state, action, action_prob, -1, i == 0)
                        elif i == 0:
                            self.dumpData(turn_, state, action, action_prob, 0, True)
                        else:
                            self.dumpData(turn_, state, action, action_prob, 0, False)
                        turn_ = 2 if turn_ == 1 else 1

                    
                    self.sampleStartState()
                    launcher = 1
                    self.env.reset(self.states[-1])

                    self.model1_score += 1
                    if self.isGameEnd():
                        self.outputScore()
                        self.model1_score = 0
                        self.model2_score = 0
                    #print(f'rally end, next launcher{launcher}')

            print(self.states[-1])
        
        self.save()
        self.finished.emit()
    