# This Python file uses the following encoding: utf-8
import sys
import os
from PySide6.QtWidgets import QFileDialog, QTableWidgetItem
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Signal, Slot, QThread, QWaitCondition, QMutex, QObject
import numpy as np
import pandas as pd
from GUI.StimulateWidget import StimulateWidget
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import matplotlib.backends.backend_agg as agg
import matplotlib.ticker as mtick
from Utils.shotTypeConvert import shotTypeEn2Zh, shotTypeZh2En
import heapq
import ast
from typing import Dict, Tuple, Optional, Literal, List
from GUI.field import Field
import seaborn as sns

ANIMATION = True


class StimulateWidgetCtrl(QObject):
    updateRegion = Signal(int, int, int)
    def __init__(self, widget: StimulateWidget):
        super().__init__(widget)
        self.__parent = widget
        self.ui = widget.ui
        self.field: Field = self.ui.field.scene()
        self.ui.nextball.clicked.connect(self.nextBall)

        # load file event
        self.ui.load_file.clicked.connect(self.loadFile)

        self.enviornment_thread = None

        self.ui.ball_round_choose.valueChanged.connect(self.updateBallRound)

        self.shot_distribution = ShotDistribution()

        self.jump = False

        # init graph drawing class
        self.lose_reason_statistic_A = LoseReasonStatistic()
        self.lose_reason_statistic_A.colors = ['aqua','blue','deepskyblue']
        self.lose_reason_statistic_B = LoseReasonStatistic()
        self.lose_reason_statistic_B.colors = ['aqua','blue','deepskyblue']
        self.final_shot_distribution = FinalShotDistribution()
        self.landing_distribution = LandingDistribution()
        self.movement_distribution = MovementDistribution()

        self.A_win_count = 0
        self.B_win_count = 0

        self.set_count = 0

        self.history_agent_name = ''
        self.predict_agent_name = ''

        self.ui.A_win_ratio.setText(f'A Win Rate: 0.0%')
        self.ui.B_win_ratio.setText(f'B Win Rate: 0.0%')

        # statistic for record output
        self.landing_count_statistic = {key: 0 for key in range(1, 11)}
        self.move_count_statistic = {key: 0 for key in range(1, 11)}
        self.type_count_statistic = {key: 0 for key in ['發短球', '長球', '推球', '殺球', '擋小球', '平球', '放小球', '挑球', '切球', '發長球', '接不到']}

        # {(state, lose_reason): count}
        self.A_lose_reason_count_statistic: Dict[Tuple[Tuple[int,int,int],str],int] = {}
        self.B_lose_reason_count_statistic: Dict[Tuple[Tuple[int,int,int],str],int] = {}
        self.last_A_lose_reason_figure = None
        self.last_B_lose_reason_figure = None

        self.ui.save.clicked.connect(self.saveStatus)

        self.is_history_lose_reason_pie_show = True

        self.history_data = None

        self.end_of_race = False # if current status is stopped and waiting for next race

        self.first = True

        self.ui.history_landing_update.clicked.connect(lambda: self.showStatisticDistributionImage((int(self.ui.statistic_player.currentText()),
                                                                                                    int(self.ui.statistic_opponent.currentText()),
                                                                                                    int(self.ui.statistic_ball.currentText()),
                                                                                                    self.ui.statistic_type.currentText().lower()),'B'))

        self.landing_point_A:Dict[Tuple[int,int,int],List[Tuple[float, float]]] = {}
        self.landing_point_B:Dict[Tuple[int,int,int],List[Tuple[float, float]]] = {}
        self.movement_point_A:Dict[Tuple[int,int,int],List[Tuple[float, float]]] = {}
        self.movement_point_B:Dict[Tuple[int,int,int],List[Tuple[float, float]]] = {}
        self.type_mapping = {'發短球': 1, '長球': 2, '推球': 3, '殺球':4, '擋小球':5, '平球':6, '放小球':7, '挑球':8, '切球':9, '發長球':10, '接不到':11} 
        self.type_zh2en = {'發短球': 'short service', '長球': 'clear', '推球': 'push', '殺球': 'smash', '擋小球': 'return net', 
                           '平球': 'drive', '放小球':'net shot', '挑球':'lob', '切球':'drop', '發長球':'long service', '接不到':'cannot reach'}
        self.shot_type_A:Dict[Tuple[int,int,int],List[int]] = {}
        self.shot_type_B:Dict[Tuple[int,int,int],List[int]] = {}
        
        for widget in [self.ui.statistic_player, self.ui.statistic_type, self.ui.statistic_opponent, self.ui.statistic_ball]:
            widget.setEditable(True)
            widget.lineEdit().setReadOnly(True)
            widget.lineEdit().setAlignment(Qt.AlignCenter)
            for i in range(widget.count()):
                widget.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)

            

    @Slot()
    def saveStatus(self):
        filename, _ = QFileDialog.getSaveFileName(self.__parent, 'Save Status', 
                                                   f'D:/CoachAI/demo/{self.agent2_name}.txt', 'text file (*.txt)')
        if len(filename) == 0:
            return
        
        # save as dict
        output_dict = {}
        output_dict['lose_reason'] = self.lose_reason_statistic_B.lose_distribution
        output_dict['landing_distribution'] = self.landing_count_statistic
        output_dict['move_distribution'] = self.move_count_statistic
        output_dict['type_distribution'] = self.type_count_statistic
        output_dict['A_win_count'] = self.A_win_count
        output_dict['B_win_count'] = self.B_win_count
        with open(filename, 'w', encoding='utf8') as output:
            output.write(str(output_dict))

    '''
    Called if someone win
    '''
    @Slot(str)
    def setWinner(self, winner:str):
        self.end_of_race = True

        self.ui.auto_nextball.setChecked(False)
        if winner == 'A':
            self.A_win_count += 1
        else:
            self.B_win_count += 1

        self.set_count += 1

        # since the inner environment won't update the winner score
        # so we need to update it by ourselves
        score_text = self.ui.score.text()
        A_score, B_score = score_text.split(' ')[1].split(':')
        A_score = int(A_score)
        B_score = int(B_score)
        if winner == "A":
            A_score += 1
        else:
            B_score += 1
        self.ui.score.setText(f'A {A_score}:{B_score} B')  

        self.ui.A_win_ratio.setText(f'A Win Rate: {self.A_win_count/(self.A_win_count+self.B_win_count):.2%}')
        self.ui.B_win_ratio.setText(f'B Win Rate: {self.B_win_count/(self.A_win_count+self.B_win_count):.2%}')

        self.setupStatisticChart()
        self.showLoseReasonStatistic('B')

    '''
    draw statistic graph
    '''
    def setupStatisticChart(self):
        print(f'set count: {self.set_count}')
        print(f'statistic\n landing:{self.landing_count_statistic}\n'
              f'type:{self.type_count_statistic}\n'
              f'move:{self.move_count_statistic}\n')

        pixmap = self.final_shot_distribution.draw(None, list(self.type_count_statistic.values()))
        self.ui.type_distribution.setPixmap(pixmap)

        history_A_win_count = self.history_data['A_win_count']
        history_B_win_count = self.history_data['B_win_count']
        self.ui.A_win_ratio.setText(f'A Win Rate: {history_A_win_count/(history_A_win_count+history_B_win_count):.2%} -> {self.A_win_count/(self.A_win_count+self.B_win_count):.2%}')
        self.ui.B_win_ratio.setText(f'B Win Rate: {history_B_win_count/(history_A_win_count+history_B_win_count):.2%} -> {self.B_win_count/(self.A_win_count+self.B_win_count):.2%}')

    @Slot(str, str, dict)
    def updateLoseReasonStatistic(self, lose_reason:str, loser:str, action:dict):
        self.updateLoseReasonPie(lose_reason, loser)

        if action is None:
            return
        
        # each item is (x, y) tuple
        #state_player, state_opponent, state_ball = action['state']
        #
        #ball_region = self.coordContinuous2Discrete(*state_ball, 1)
        #opponent_region = self.coordContinuous2Discrete(*state_opponent, 1)
        #player_region = self.coordContinuous2Discrete(*state_player, 2)
        #
        #state = player_region, opponent_region, ball_region

        state = (*action['state'], action['last_type'])

        if loser == 'A':
            if (state, lose_reason) not in self.A_lose_reason_count_statistic:
                self.A_lose_reason_count_statistic[(state, lose_reason)] = 0
            self.A_lose_reason_count_statistic[(state, lose_reason)] += 1

        elif loser =='B':
            if (state, lose_reason) not in self.B_lose_reason_count_statistic:
                self.B_lose_reason_count_statistic[(state, lose_reason)] = 0
            self.B_lose_reason_count_statistic[(state, lose_reason)] += 1

        #lose reason statistic
        self.showLoseReasonStatistic(loser)

    def showLoseReasonStatistic(self, player:str):
        if player == 'A':
            # Get the three largest values with their keys
            top4 = heapq.nlargest(4, self.A_lose_reason_count_statistic.items(), key=lambda x: x[1])
            total = sum(self.A_lose_reason_count_statistic.values())
        elif player =='B':
            # Get the three largest values with their keys
            top4 = heapq.nlargest(4, self.B_lose_reason_count_statistic.items(), key=lambda x: x[1])
            total = sum(self.B_lose_reason_count_statistic.values())
        else:
            return

        for rank, ((state, lose_reason), count) in enumerate(top4):
            for i in range(3):
                item = QTableWidgetItem(f'{state[i]}')
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.ui.lose_state_table.setItem(i, rank, item)
            item = QTableWidgetItem(f'{lose_reason.title()}')
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.lose_state_table.setItem(3, rank, item)
            item = QTableWidgetItem(f'{count/total:.2%}')
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.lose_state_table.setItem(4, rank, item)

        if len(top4) > 0:
            top1 = top4[0]
            state_player, state_opponent, state_ball, state_shot = top1[0][0]
            self.ui.statistic_player.setCurrentIndex(state_player-1)
            self.ui.statistic_opponent.setCurrentIndex(state_opponent-1)
            self.ui.statistic_ball.setCurrentIndex(state_ball-1)
            self.ui.statistic_type.setCurrentText(state_shot.title())

            self.showStatisticDistributionImage(top1[0][0], player)


    def updateLoseReasonPie(self, lose_reason: str, loser: str):
        score_text = self.ui.score.text()
        A_score, B_score = score_text.split(' ')[1].split(':')

        if loser == 'A':
            self.lose_reason_statistic_A.addDataPoint(lose_reason)
            pixmap = self.lose_reason_statistic_A.draw()  
            self.last_A_lose_reason_figure = pixmap      
        elif loser == 'B':
            self.lose_reason_statistic_B.addDataPoint(lose_reason)
            pixmap = self.lose_reason_statistic_B.draw()  
            self.last_B_lose_reason_figure = pixmap

        #self.ui.lose_reason_pie.setPixmap(pixmap)

    def showStatisticDistributionImage(self, state, current_player: str):        
        if current_player == 'A':
            if state in self.landing_point_A:
                coords = np.array(self.landing_point_A[state])
            else:
                coords = np.array(self.landing_point_B[state])
        else:
            if state in self.landing_point_B:
                coords = np.array(self.landing_point_B[state])
            else:
                coords = np.array(self.landing_point_A[state])

        if current_player == 'A':
            self.landing_distribution.predict_agent_name = self.agent1_name
            self.movement_distribution.predict_agent_name = self.agent1_name
            self.shot_distribution.predict_agent_name = self.agent1_name
        else:
            self.landing_distribution.predict_agent_name = self.agent2_name
            self.movement_distribution.predict_agent_name = self.agent2_name
            self.shot_distribution.predict_agent_name = self.agent2_name
        pixmap = self.landing_distribution.draw(coords)

        w = self.ui.history_landing_image.width()
        h = self.ui.history_landing_image.height()

        self.ui.history_landing_image.setPixmap(pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if current_player == 'A':
            if state in self.movement_point_A:
                coords = np.array(self.movement_point_A[state])
            else:
                coords = np.array(self.movement_point_B[state])
        else:
            if state in self.movement_point_B:
                coords = np.array(self.movement_point_B[state])
            else:
                coords = np.array(self.movement_point_A[state])
        pixmap = self.movement_distribution.draw(coords)

        w = self.ui.history_landing_image.width()
        h = self.ui.history_landing_image.height()

        self.ui.history_movement_image.setPixmap(pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if self.end_of_race:
            # history shot type
            counts = self.shot_type_A[state][:] # copy
            total = sum(counts)
            for i in range(len(counts)):
                counts[i] /= total
            self.shot_distribution.title = 'History Shot Distribution'
            self.setTypeDistributionImage(counts)

    # event function for csv loading
    @Slot()
    def loadFile(self):
        filename, _ = QFileDialog.getOpenFileName(self.__parent, caption="Open file", dir="D:/CoachAI/demo/", filter="csv file(*.csv)")
        if len(filename) == 0:
            return
        
        self.ui.ball_round_id.setText("0")
        self.ui.ball_round_choose.setEnabled(False)
        self.ui.ball_round_choose.setValue(0)
        self.ui.load_file.setEnabled(False)
        self.ui.load_file.setText('loading...')
        if self.enviornment_thread is not None and not self.enviornment_thread.isFinished():
            self.enviornment_thread.wakeUp()
            self.enviornment_thread.terminate()
        
        self.ui.filename.setText(filename)

        agent1Name, agent2Name = filename.split('/')[-1].rsplit('.',1)[0].rsplit('_',1)[0].split('vs', 1)
        self.agent1_name = agent1Name
        self.agent2_name = agent2Name
        self.final_shot_distribution.predict_agent_name = agent2Name
        self.shot_distribution.predict_agent_name = agent2Name
        self.lose_reason_statistic_B.agent_name = agent2Name
        self.lose_reason_statistic_A.agent_name = agent1Name
        self.predict_agent_name = agent2Name

        self.enviornment_thread = EnviornmentThread(filename)
        self.enviornment_thread.paused.connect(self.currentBallFinished)
        self.enviornment_thread.makeEnvFinished.connect(self.threadLoadFinished)
        self.enviornment_thread.start()

        # statistic landing pos of each state
        data = pd.read_csv(filename)

        data_A = data[data['obs_player'] == 'A']
        data_B = data[data['obs_player'] == 'B']

        for landing_point, data in [(self.landing_point_A, data_A), (self.landing_point_B, data_B)]:
            groups = data.groupby(['rally'])
            for name, group in groups:
                for i, (id, row) in enumerate(group.iterrows()):
                    state = ast.literal_eval(row['state'])
                    if i == 0:
                        last_type = 'receiving'
                    else:
                        last_type = self.type_zh2en[group['act_ball_type'].iloc[i-1]]
                    state = (*state, last_type)
                    landing_x = row['act_landing_location_x']
                    landing_y = row['act_landing_location_y']

                    if not state in landing_point:
                        landing_point[state] = []
                    landing_point[state].append((landing_x, landing_y))

        for movement_point, data in [(self.movement_point_A, data_A), (self.movement_point_B, data_B)]:
            groups = data.groupby(['rally'])
            for name, group in groups:
                for i, (id, row) in enumerate(group.iterrows()):
                    state = ast.literal_eval(row['state'])
                    if i == 0:
                        last_type = 'receiving'
                    else:
                        last_type = self.type_zh2en[group['act_ball_type'].iloc[i-1]]
                    state = (*state, last_type)
                    movement_x = row['act_player_defend_x']
                    movement_y = row['act_player_defend_y']
                    if movement_y > 0:
                        movement_y = -movement_y

                    if not state in movement_point:
                        movement_point[state] = []
                    movement_point[state].append((movement_x, movement_y))

        for shot_type, data in [(self.shot_type_A, data_A), (self.shot_type_B, data_B)]:
            groups = data.groupby(['rally'])
            for name, group in groups:
                for i, (id, row) in enumerate(group.iterrows()):
                    state = ast.literal_eval(row['state'])
                    if i == 0:
                        last_type = 'receiving'
                    else:
                        last_type = self.type_zh2en[group['act_ball_type'].iloc[i-1]]
                    state = (*state, last_type)
                    type = row['act_ball_type']

                    if not state in shot_type:
                        shot_type[state] = [0] * 11
                    shot_type[state][self.type_mapping[type]-1] += 1
        
    
    @Slot(dict, str)
    def updateAction(self, action:dict, player: str): # player is 'A' or 'B'
        if action['land_prob'] is not None and action['move_prob'] is not None:
            self.field.setDistributionImg(True,  action['land_prob'], action['move_prob'])
            #self.distribution_image.set_color(1,1,1,1)

        current_agent_name = self.agent1_name if player == 'A' else self.agent2_name

        if action['shot_prob'] is not None:
            self.shot_distribution.setCurrentAgentName(current_agent_name)

            if len(action['shot_prob']) > 0:
                self.setTypeDistributionImage(action['shot_prob'])
            else:
                self.setTypeDistributionImage([0]*11)

        #statistic B only
        if player == 'B':
            #self.landing_count_statistic[action['landing_region']] += 1
            #self.move_count_statistic[action['opponent_region']] += 1
            self.type_count_statistic[action['ball_type_name']] += 1

        if player == 'A':
            if self.last_A_lose_reason_figure is not None:
                self.ui.lose_reason_pie.setPixmap(self.last_A_lose_reason_figure)
            else:
                self.ui.lose_reason_pie.clear()
        elif player == 'B':
            if self.last_B_lose_reason_figure is not None:
                self.ui.lose_reason_pie.setPixmap(self.last_B_lose_reason_figure)
            else:
                self.ui.lose_reason_pie.clear()

        self.showLoseReasonStatistic(player)

    def setTypeDistributionImage(self, probs: list):
        pixmap = self.shot_distribution.draw(probs)     
        self.ui.type_distribution.setPixmap(pixmap)

    # init csv loading thread function
    @Slot()
    def threadLoadFinished(self):
        # Bind slot function for enviornment to change GUI
        self.enviornment_thread.env.updateAction.connect(self.updateAction)
        #self.enviornment_thread.env.setLoseReasonDistributionImage.connect(self.setLoseReasonDistributionImage)
        self.enviornment_thread.env.updateLoseStatistic.connect(self.updateLoseReasonStatistic)
        self.enviornment_thread.env.setScore.connect(self.setScore)
        self.enviornment_thread.env.setStrokeType.connect(self.setStrokeType)
        self.enviornment_thread.env.setBallPos.connect(self.setBallPos)
        self.enviornment_thread.env.setBallScale.connect(self.field.setBallScale)
        self.enviornment_thread.env.setPlayerAPos.connect(self.setPlayerAPos)
        self.enviornment_thread.env.setPlayerBPos.connect(self.setPlayerBPos)
        self.enviornment_thread.env.setBallRound.connect(self.setBallRound)
        self.enviornment_thread.env.setSetWinner.connect(self.setWinner)
        self.enviornment_thread.changeBallLauncher.connect(self.changeBallLauncher)

        self.ui.load_file.setEnabled(True)
        self.ui.load_file.setText('import')
        self.ui.ball_round_choose.setEnabled(True)
        self.ui.ball_round_choose.setRange(0, self.enviornment_thread.total_ball_count-1)
        self.ui.ball_round_choose.setValue(0)  

    '''
    input (x,y), convert to region(1~10)
    position 1 means upper field, 2 means down field
    '''
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

    @Slot(str)
    def changeBallLauncher(self, player: str):
        self.field.changeBallLauncher(player)  
        
    @Slot(int)
    def updateBallRound(self, round: int):
        #if int(self.ui.ball_round.text()) != round:
        #    self.jump = True
        self.ui.ball_round_id.setText(f'{round+1}')

    @Slot(float, float)
    def setPlayerAPos(self, x: float, y: float):
        self.ui.playerA_x.setText(f'{x:.1f}')
        self.ui.playerA_y.setText(f'{y:.1f}')
        self.field.setPlayerAPos(x, y)

    @Slot(float, float)
    def setPlayerBPos(self, x: float, y: float):
        self.ui.playerB_x.setText(f'{x:.1f}')
        self.ui.playerB_y.setText(f'{y:.1f}')
        self.field.setPlayerBPos(x, y)

    @Slot(float, float)
    def setBallPos(self, x: float, y: float):
        self.ui.landing_x.setText(f'{x:.1f}')
        self.ui.landing_y.setText(f'{y:.1f}')
        self.field.setBallPos(x, y)

    # event function to activate when enviornment finish a round 
    @Slot()
    def currentBallFinished(self):
        self.ui.nextball.setDisabled(False)
        self.ui.ball_round_choose.setValue(self.ui.ball_round_choose.value()+1)
        # auto click
        if self.ui.auto_nextball.isChecked():
            self.ui.nextball.animateClick()

    @Slot(int)
    def setBallRound(self, round:int):
        self.ui.ball_round.setText(f'Round: {round}')

    @Slot(str, str)
    def setStrokeType(self, player:str, type:str):
        if type is None or type == "":
            type = ""
        else:
            type = shotTypeZh2En(type)

        print(type)
        if player == 'A':
            self.ui.playerA_type.setText(type)
        elif player == 'B':
            self.ui.playerB_type.setText(type)
        else:
            raise NotImplementedError

    @Slot(int, int)
    def setScore(self, playerA_score: int, playerB_score: int):
        self.ui.score.setText(f'A {playerA_score}:{playerB_score} B')

    @Slot()
    def nextBall(self):
        self.end_of_race = False
        self.ui.nextball.setDisabled(True)

        self.enviornment_thread.index = self.ui.ball_round_choose.value()
        self.enviornment_thread.jump = self.jump
        self.enviornment_thread.wakeUp()

class EnviornmentThread(QThread):
    paused = Signal()
    makeEnvFinished = Signal()
    changeBallLauncher = Signal(str)
    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.filepath = filepath

        self.mutex = QMutex()
        self.condition = QWaitCondition()

        self.total_ball_count = 147

        self.index = 0
        self.jump = False

    def pre_run(self):
        from StrategicEnvironment.multiagent.policy import InteractivePolicy, Player_Policy, Policy
        import StrategicEnvironment.multiagent.scenarios as scenarios
        from StrategicEnvironment.multiagent.environment import MultiAgentEnv

        print('make enviornment...')
        # initial environment
        # load scenario and bulid world
        scenario = scenarios.load('badminton.py').Scenario()
        world=scenario.make_world(resistance_force_n = 1)

        world.discrete_court = False
        world.decide_defend_location = False
        world.number_of_match = 1# total number of match
        world.player = 1 # B

        #world.ball.returnable_distance = 100
        #world.ball.returnable_height = 150L

        # make environment for current world
        #self.env = make_env('badminton')
        self.env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                                 info_callback=None, top_viewer = False, side_viewer = False, show_detail=False)
        
        #self.env.render()

        policies = []
        policies.append(Player_Policy(self.env, 0, self.filepath,'A'))
        policies.append(Player_Policy(self.env,1, self.filepath,'B'))

        obs_n = self.env.reset()

        try:
            while True:
                while True:
                    act_n = []
                    for i, policy in enumerate(policies):
                        act_n.append(policy.action(obs_n[i]))
                    obs_n, reward_n, done_n, info_n = self.env.step(act_n)
                    if done_n[0].rally_done:
                        self.env.reset()
                        break 
                    
                if done_n[0].match_done:
                    break
        except:
            pass

        self.env.save_match_data('input1.csv')

    def run(self) :
        print('load module...')
        # from make_env import make_env
        from StrategicEnvironment.multiagent.policy import InteractivePolicy, Player_Policy, Policy
        import StrategicEnvironment.multiagent.scenarios as scenarios
        from StrategicEnvironment.multiagent.environment import MultiAgentEnv
        
        print('make enviornment...')
        # initial environment
        # load scenario and bulid world
        scenario = scenarios.load('badminton.py').Scenario()
        world=scenario.make_world(resistance_force_n = 1)

        world.discrete_court = False
        world.decide_defend_location = False 
        world.number_of_match = 1# total number of match
        world.player = 1 # B

        #world.ball.returnable_distance = 100
        #world.ball.returnable_height = 150L

        # make environment for current world
        #self.env = make_env('badminton')
        self.env = MultiAgentEnv(world, self.filepath,
                                 scenario.reset_world, scenario.reward, scenario.observation,
                                 info_callback=None, top_viewer = ANIMATION, side_viewer = False)
        
        import debugpy
        debugpy.debug_this_thread()

        # set policy for each agent
        #self.env.render()

        self.policies = []
        self.policies.append(Player_Policy(self.env,0, self.filepath,'A'))
        self.policies.append(Player_Policy(self.env,1, self.filepath,'B'))

        self.obs_n = self.env.reset()

        # waiting
        self.paused.emit()
        self.makeEnvFinished.emit()
        self.mutex.lock()
        self.condition.wait(self.mutex)
        self.mutex.unlock()

        # for i, policy in enumerate(self.policies):
        #     act_n.append(policy.action(self.obs_n[i]))
        #match_index = 0

        
        while True:
            act_n = []
            #act_n.append(self.policies.action(match_index))
            #print(self.jump)
            if self.jump:
                for i, policy in enumerate(self.policies):
                    act_n.append(policy.jump(self.index))
                self.obs_n, reward_n, done_n, info_n = self.env.step(act_n, self.jump, self.index)
                self.jump = False
            act_n = []
            for i, policy in enumerate(self.policies):
                #print(i)
                #print(self.obs_n[i])
                act_n.append(policy.action(self.obs_n[i]))

            if act_n[0]['player_location'][0] is None: # B launcher
                self.changeBallLauncher.emit('A')
            else:
                self.changeBallLauncher.emit('B') 
            self.obs_n, reward_n, done_n, info_n = self.env.step(act_n, self.jump, self.index)

            #print(act_n)
            #print(self.obs_n)
            if done_n[0].rally_done:
                self.env.reset()

            if self.isInterruptionRequested():
                break

            # waiting for user to start next round when finished a round
            self.paused.emit()
            self.mutex.lock()
            self.condition.wait(self.mutex)
            self.mutex.unlock()             
            
            if self.isInterruptionRequested():
                break
            #match_index += 1

    #unlock mutex to start next ball
    def wakeUp(self):
        self.mutex.lock()
        self.condition.wakeAll()
        self.mutex.unlock()

class VisualizeGraph:
    def __init__(self):
        pass

    def image2pixmap(self, image:np.ndarray):
        if image is None:
            return QPixmap()
        qimage = QImage(image.data, image.shape[1], image.shape[0], 
                image.strides[0], QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimage)
    
    def draw2pixmap(self, fig:plt.Figure, rotate = None):
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        img = np.frombuffer(raw_data, dtype=np.uint8).reshape(size[::-1] + (3,))

        if rotate is not None:
            img = np.ascontiguousarray(np.rot90(img, axes=(1,0)))

        pixmap = self.image2pixmap(img)

        return pixmap

"""
draw shot type statistic as bar chart and compare with history data
"""
class FinalShotDistribution(VisualizeGraph):
    def __init__(self):
        super().__init__()

        self.history_agent_name = 'history'
        self.predict_agent_name = 'predict'

    def draw(self, history_probs: np.ndarray, probs: np.ndarray):
        probs /= np.sum(probs)
        history_probs /= np.sum(history_probs)

        # 切換中文字型: 微軟正黑體
        matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'

        # Create a figure and axes
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.15, bottom=0.2, right=0.99, top=0.9, wspace=0, hspace=0)
        # Plot the probabilities using a bar graph
        bars = ax.bar(range(len(probs)), probs, alpha=0.75, label=self.predict_agent_name)
        if history_probs is not None:
            bars_standard = ax.bar(range(len(history_probs)), history_probs, alpha=0.75, label=self.history_agent_name)
        ax.legend(fontsize=14)
        ax.set_title("Shot Type Distribution", fontsize=17)

        # Add labels to the plot
        ax.set_xlabel('Type')
        ax.set_ylabel('Probability',fontsize=13)

        # Set the x-axis tick labels to correspond to the events
        # events = ['發短球', '長球', '推撲球', '殺球', '接殺防守', '平球', '網前球', '挑球', '切球', '發長球']
        events = ['short service', 'clear', 'push/rush', 'smash', 'defensive shot', 'drive', 'net shot','lob', 'drop', 'long service', 'cannot reach']
        ax.set_xticks(range(len(probs)))
        ax.set_xticklabels(events, fontsize=15, rotation=30, ha='right')
        ax.tick_params(axis='x', pad=-5) 
        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylim(ymin=0.)

        # Add the values on top of the bars
        for bar in bars:
            height = bar.get_height()
            text = f'{height:.2%}' if height != 0 else '--%'
            ax.text(bar.get_x() + bar.get_width() / 2, height, text, ha='center', va='bottom', fontsize =11)

        pixmap = self.draw2pixmap(fig)
        plt.close()
        return pixmap

"""
draw shot type statistic as bar chart
"""
class ShotDistribution(VisualizeGraph):
    def __init__(self):
        super().__init__()
        self.history_agent_name = 'history'
        self.predict_agent_name = 'ShuttleNet'

        self.title = "Shot Type Distribution"

    def setCurrentAgentName(self, name:str):
        self.predict_agent_name = name

    # generate the shot landing position distribution using plt 
    # output the image in the form of array
    def draw(self, probs: list):
        # 切換中文字型: 微軟正黑體
        matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'

        # Create a figure and axes
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.15, bottom=0.2, right=0.99, top=0.9, wspace=0, hspace=0)

        # Plot the probabilities using a bar graph
        bars = ax.bar(range(len(probs)), probs, alpha=0.75, label =self.predict_agent_name)
        ax.legend(fontsize=14)
        ax.set_title(self.title, fontsize=17)

        # Add labels to the plot
        ax.set_xlabel('Type')
        ax.set_ylabel('Probability')

        # Set the x-axis tick labels to correspond to the events
        # events = ['發短球', '長球', '推撲球', '殺球', '接殺防守', '平球', '網前球', '挑球', '切球', '發長球']
        events = ['short service', 'clear', 'push/rush', 'smash', 'defensive shot', 'drive', 'net shot','lob', 'drop', 'long service', 'cannot reach']
        ax.set_xticks(range(len(probs)))
        ax.set_xticklabels(events, fontsize=15, rotation=30, ha='right')
        ax.tick_params(axis='x', pad=-5) 
        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylim(ymin=0.)

        # Add the values on top of the bars
        for bar in bars:
            height = bar.get_height()
            text = f'{height:.2%}' if height != 0 else '--%'
            ax.text(bar.get_x() + bar.get_width() / 2, height, text, ha='center', va='bottom')

        pixmap = self.draw2pixmap(fig)
        plt.close()
        return pixmap
    
class LandingDistribution(VisualizeGraph):
    def __init__(self):
        super().__init__()
        self.history_agent_name = 'history'
        self.predict_agent_name = 'predict'

    def setCurrentAgentName(self, name:str):
        self.predict_agent_name = name

    # generate the shot landing position distribution using plt 
    # output the image in the form of array
    def draw(self, coords: np.ndarray):
        # 切換中文字型: 微軟正黑體
        matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(4,4))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-200, 200)
        ax.set_ylim(-70, 400)

        x = 177.5
        y = 480

        plt.plot([25-x, 330-x], [810-y, 810-y], color='black', linestyle='-', linewidth=1.5)
        plt.plot([25-x, 330-x], [756-y, 756-y], color='black', linestyle='-', linewidth=1.5)
        plt.plot([25-x, 330-x], [594-y, 594-y], color='black', linestyle='-', linewidth=1.5)
        #plt.plot([25-x, 330-x], [366-y, 366-y], color='black', linestyle='-', linewidth=1)
        #plt.plot([25-x, 330-x], [204-y, 204-y], color='black', linestyle='-', linewidth=1)
        #plt.plot([25-x, 330-x], [150-y, 150-y], color='black', linestyle='-', linewidth=1)
        #plt.plot([25-x, 25-x],  [150-y, 810-y], color='black', linestyle='-', linewidth=1)
        plt.plot([25-x, 25-x],  [480-y, 810-y], color='black', linestyle='-', linewidth=1.5)
        #plt.plot([50-x, 50-x],  [150-y, 810-y], color='black', linestyle='-', linewidth=1)
        plt.plot([50-x, 50-x],  [480-y, 810-y], color='black', linestyle='-', linewidth=1.5)
        #plt.plot([177.5-x, 177.5-x], [150-y, 810-y], color='black', linestyle='-', linewidth=1)
        plt.plot([177.5-x, 177.5-x], [594-y, 810-y], color='black', linestyle='-', linewidth=1.5)
        #plt.plot([305-x, 305-x], [150-y, 810-y], color='black', linestyle='-', linewidth=1)
        plt.plot([305-x, 305-x], [480-y, 810-y], color='black', linestyle='-', linewidth=1.5)
        #plt.plot([330-x, 330-x], [150-y, 810-y], color='black', linestyle='-', linewidth=1)
        plt.plot([330-x, 330-x], [480-y, 810-y], color='black', linestyle='-', linewidth=1.5)
        plt.plot([25-x, 330-x],  [480-y, 480-y], color='black', linestyle='-', linewidth=1.5) 

        sns.kdeplot(x=coords[:,0], y = coords[:,1], color='orange')

        pixmap = self.draw2pixmap(fig)
        plt.close()
        return pixmap
    
class MovementDistribution(VisualizeGraph):
    def __init__(self):
        super().__init__()
        self.history_agent_name = 'history'
        self.predict_agent_name = 'predict'

    def setCurrentAgentName(self, name:str):
        self.predict_agent_name = name

    # generate the shot landing position distribution using plt 
    # output the image in the form of array
    def draw(self, coords: np.ndarray):
        # 切換中文字型: 微軟正黑體
        matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(4,4))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-200, 200)
        ax.set_ylim(-400, 70)

        x = 177.5
        y = 480

        #plt.plot([25-x, 330-x], [810-y, 810-y], color='w', linestyle='-', linewidth=1.5)
        #plt.plot([25-x, 330-x], [756-y, 756-y], color='w', linestyle='-', linewidth=1.5)
        #plt.plot([25-x, 330-x], [594-y, 594-y], color='w', linestyle='-', linewidth=1.5)
        plt.plot([25-x, 330-x], [366-y, 366-y], color='black', linestyle='-', linewidth=1)
        plt.plot([25-x, 330-x], [204-y, 204-y], color='black', linestyle='-', linewidth=1)
        plt.plot([25-x, 330-x], [150-y, 150-y], color='black', linestyle='-', linewidth=1)
        #plt.plot([25-x, 25-x],  [150-y, 810-y], color='black', linestyle='-', linewidth=1)
        plt.plot([25-x, 25-x],  [480-y, 150-y], color='black', linestyle='-', linewidth=1.5)
        #plt.plot([50-x, 50-x],  [150-y, 810-y], color='black', linestyle='-', linewidth=1)
        plt.plot([50-x, 50-x],  [480-y, 150-y], color='black', linestyle='-', linewidth=1.5)
        #plt.plot([177.5-x, 177.5-x], [150-y, 810-y], color='black', linestyle='-', linewidth=1)
        plt.plot([177.5-x, 177.5-x], [366-y, 150-y], color='black', linestyle='-', linewidth=1.5)
        #plt.plot([305-x, 305-x], [150-y, 810-y], color='black', linestyle='-', linewidth=1)
        plt.plot([305-x, 305-x], [480-y, 150-y], color='black', linestyle='-', linewidth=1.5)
        #plt.plot([330-x, 330-x], [150-y, 810-y], color='black', linestyle='-', linewidth=1)
        plt.plot([330-x, 330-x], [480-y, 150-y], color='black', linestyle='-', linewidth=1.5)
        plt.plot([25-x, 330-x],  [480-y, 480-y], color='black', linestyle='-', linewidth=1.5)  

        sns.kdeplot(x=coords[:,0], y = coords[:,1], color='orange')

        pixmap = self.draw2pixmap(fig)
        plt.close()
        return pixmap

"""
draw region statistic as radar chart
"""
class RegionStatistic(VisualizeGraph):
    def __init__(self, title:str, region_label:str, x_label:str, y_label:str):
        super().__init__()
        self.title = title
        self.current_region_percent = None
        self.history_region_percent = None
        self.region_name = [f'{i}' for i in range(1,11)]
        self.region_name.append('')
        self.x_label = x_label
        self.y_label = y_label
        self.region_label = region_label
        self.history_agent_name = 'history'
        self.predict_agent_name = 'predict'

    def setCurrentAgentName(self, name:str):
        self.predict_agent_name = name

    def setCurrentDistribution(self, prob:list):
        self.current_region_percent = np.array(prob) / sum(prob)

    def setHistoryDistribution(self, prob:Optional[list]):
        if prob is None:
            self.history_region_percent = None
        else:
            self.history_region_percent = np.array(prob) / sum(prob)

    def draw(self):
        # Convert probabilities to radians for radar chart
        angles = np.linspace(0, 2 * np.pi, len(self.current_region_percent)+1).tolist()

        # make radar line close
        current_region_percent = np.append(self.current_region_percent, self.current_region_percent[0])
        if self.history_region_percent is not None:
            history_region_percent = np.append(self.history_region_percent, self.history_region_percent[0])

        fig, ax = plt.subplots(figsize=(5,5),subplot_kw={'projection': 'polar'})
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.88, wspace=0, hspace=0)

        # Plot the probabilities using a radar chart
        ax.plot(angles, current_region_percent*100, linewidth=1, alpha=0.75, label=self.predict_agent_name)
        ax.fill(angles, current_region_percent*100, alpha=0.25)

        if self.history_region_percent is not None:
            ax.plot(angles, history_region_percent*100,  linewidth=1, alpha=0.75, label=self.history_agent_name)
            ax.fill(angles, history_region_percent*100, alpha=0.25)

        ax.legend(fontsize=17)
        ax.set_title(self.title, fontsize=25)

        # Set the x-axis tick labels to correspond to the events
        ax.set_xticks(angles)
        ax.set_xticklabels(self.region_name, fontsize=18)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        
        # Set the radial axis range to match the maximum probability
        #ax.set_ylim(0, max(probs))

        # Add the values on top of the data points
        #for angle, prob in zip(angles, probs):
        #    ax.text(angle, prob, f'{prob:.0%}', ha='center', va='bottom')

        pixmap = self.draw2pixmap(fig)
        plt.close()
        return pixmap

"""
draw lose reason statistic as pie chart
"""
class LoseReasonStatistic(VisualizeGraph):
    def __init__(self):
        super().__init__()
        self.lose_distribution = {}
        self.agent_name = ''
        self.colors = ['aqua','blue','deepskyblue']
        self.lose_distribution['Return Fail'] = 1
        self.lose_distribution['Outside'] = 1

    def addDataPoint(self, lose_reason: str):
        if lose_reason != 'outside':
            lose_reason = 'Return Fail'
        else:
            lose_reason = 'Outside'
        if lose_reason not in self.lose_distribution:
            self.lose_distribution[lose_reason] = 0

        self.lose_distribution[lose_reason] += 1

    def draw(self):
        percentage = []
        labels = []
        total = sum(self.lose_distribution.values())
        for key, value in self.lose_distribution.items():
            percentage.append(value / total)
            labels.append(key)

        fig, ax = plt.subplots(figsize=(4, 16/3), dpi=80)
        plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, wspace=0, hspace=0)
        pies = ax.pie(percentage, colors=self.colors)
                      #,textprops={'fontsize': 20}, labeldistance=0.3)
        #pies = ax.pie([23,700], #labels=['outside','return fail'],  #[766,333]
        #          colors=['aqua','deepskyblue','blue'],)#textprops={'fontsize': 25}, labeldistance=0.3)
        #for text in pies[1]:
        #    x, y = text.get_position()
        #    text.set_position((x, y + 0.1))
        ax.set_title(f"{self.agent_name}\nLose Reason Statistic", fontsize=25)
        ax.legend(labels=labels,fontsize=30,loc='lower left', bbox_to_anchor=(-0.3, -0.55))
    
        pixmap = self.draw2pixmap(fig)
        plt.close()
        return pixmap