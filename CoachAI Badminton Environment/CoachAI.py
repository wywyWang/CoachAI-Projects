import pandas as pd
from typing import Tuple
import numpy as np
import pickle
import tqdm
from RLEnvironment import Env
import torch
from a2c import A2C

class GenerateThread:
    def __init__(self):
        self.shottype_mapping = ['發短球', '長球', '推球', '殺球', '擋小球', '平球', '放小球', '挑球', '切球', '發長球', '接不到']
        self.round = 1
        self.rally = 0
        self.output = pd.DataFrame()
        self.model1_score = 0
        self.model2_score = 0
    
    # convert action, stage based data to rally base
    def dumpData(self, player:int, state: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]], 
                 action: Tuple[int, Tuple[float, float], Tuple[float, float]], 
                 action_prob:Tuple[list, Tuple[list, list, list], Tuple[list, list, list]], reward: int, is_launch: bool):
        # launch failed
        #if is_launch and reward == -1:
        #    return
        state_player, state_opponent, state_ball = state
        action_type, action_land, action_move = action
        action_type_prob, action_land_gmm_param, action_move_gmm_param = action_prob

        if is_launch:
            self.rally += 1

        # 11 mean cannot reach, prev state is last state
        #if action_type != 11:
        player_x, player_y = action_move
        landing_x, landing_y = action_land
        row = pd.DataFrame([{'rally':self.rally,
                            'obs_ball_round':self.round,
                            'obs_player': 'A' if player == 1 else 'B',
                            'A_score': self.model1_score,
                            'B_score': self.model2_score,
                            'obs_serve': is_launch,
                            'act_landing_location_x':landing_x,
                            'act_landing_location_y':landing_y,
                            'act_player_defend_x': player_x,
                            'act_player_defend_y':player_y,
                            'shot_prob': action_type_prob,
                            'land_prob': action_land_gmm_param,
                            'move_prob': action_move_gmm_param,
                            'act_ball_type':self.shottype_mapping[int(action_type)-1]}])
        self.output = pd.concat([self.output, row])
        
        self.round += 1

        # current rally end
        if reward == -1:
            self.round = 1

    def close(self, output_filename='output.csv'):
        self.output.to_csv(output_filename, index=False)
# episodes, model_path, side=2, opponent="Anthony Sinisuka GINTING"
from SuperviseAgent import SuperviseAgent
class badminton_env(GenerateThread):
    def __init__(self, rally_count:int, side: int, opponent: str):
        super().__init__()
        self.cnt = 0
        self.all_action = []
        self.rally_actions = []
        self.rally_count = rally_count
        self.env = Env()

        opponent_table=['Kento MOMOTA', 'CHOU Tien Chen', 'Anthony Sinisuka GINTING', 'CHEN Long', 'CHEN Yufei', 'TAI Tzu Ying', 'Viktor AXELSEN', 'Anders ANTONSEN', 'PUSARLA V. Sindhu', 'WANG Tzu Wei', 'Khosit PHETPRADAB', 'Jonatan CHRISTIE', 'NG Ka Long Angus', 'SHI Yuqi', 'Ratchanok INTANON', 'An Se Young', 'Busanan ONGBAMRUNGPHAN', 'Mia BLICHFELDT', 'LEE Zii Jia', 'LEE Cheuk Yiu', 'Rasmus GEMKE', 'Michelle LI', 'Supanida KATETHONG', 'Carolina MARIN', 'Pornpawee CHOCHUWONG', 'Sameer VERMA', 'Neslihan YIGIT', 'Hans-Kristian Solberg VITTINGHUS', 'LIEW Daren', 'Evgeniya KOSETSKAYA', 'KIDAMBI Srikanth', 'Soniia CHEAH', 'Gregoria Mariska TUNJUNG', 'Akane YAMAGUCHI', 'HE Bingjiao', '胡佑齊', '張允澤', '許喆宇', '陳政佑', '林祐賢', '李佳豪', 'LOH Kean Yew', 'Lakshya SEN', 'Kunlavut VITIDSARN', 'WANG Hong Yang', 'Kodai NARAOKA', 'JEON Hyeok Jin', 'Wen Chi Hsu', 'Nozomi Okuhara', 'WANG Zhi Yi', 'PRANNOY H. S.', 'Chico Aura DWI WARDOYO', 'LU Guang Zu', 'ZHAO Jun Peng', 'Kenta NISHIMOTO', 'NG Tze Yong', 'Victor SVENDSEN', 'WENG Hong Yang', 'Aakarshi KASHYAP', 'LI Shi Feng', 'KIM Ga Eun', 'HAN Yue', 'Other', 'NYCU']
        
        self.model = SuperviseAgent(opponent_table.index(opponent), 1)
        

        self.user_side = side
        if side == 1:
            self.user_launch = True
        else:
            self.user_launch = False

        # get first n ball from history data
        # needed ball round + 1 and filtered out last row so that last ball is not end
        # if self.is_model1_shuttleNet or self.is_model2_shuttleNet:
        print('contain ShuttleNet')
        self.init_row_count = 2 # for shuttleNet
        # else:
            # self.init_row_count = 2
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
    def reset(self):
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

        if not self.user_launch:
            self.states, reward, done, launch = self.shuttleNetDyMF_shot(True)
            if done:
                self.user_launch = True
                return self.reset()
            return self.states, launch

        return self.states, self.user_launch

    def outputScore(self):
        self.cnt += 1

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
    
    def step(self, action, launch):
        done = False
        self.states, reward, done, launch = self.usermodel_shot(action, launch)
        if done:
            self.user_launch = False
            return self.states, reward, done, launch
        
        self.states, _, done, launch = self.shuttleNetDyMF_shot(launch)
        if done:
            self.user_launch = True

        return self.states, reward, done, launch 
        # self.save()

    def usermodel_shot(self, action, launch):
        done = False
        action, action_prob = action
        state, reward = self.env.step(action, launch)

        if reward != -1:
            self.states.append(state)
        if action is not None:
            self.actions.append(action)
            self.actions_prob.append(action_prob)

        launch = False

        # round end
        if reward == -1:
            launch = True
            done = True
            self.all_action.append(self.rally_actions)
            self.rally_actions = []
            turn_ = 2 if self.user_side == 1 else 1

            # output data to dataFrame
            assert len(self.states) == len(self.actions) and  len(self.actions) == len(self.actions_prob)
            for i, (state, action, action_prob) in enumerate(zip(self.states, self.actions, self.actions_prob)):
                if i == len(self.states) - 1:
                    self.dumpData(turn_, state, action, action_prob, -1, i == 0) # last rally of round reward is -1
                elif i == 0:
                    self.dumpData(turn_, state, action, action_prob, 0, True)
                else:
                    self.dumpData(turn_, state, action, action_prob, 0, False)
                turn_ = 2 if turn_ == 1 else 1
            
            # _, _ = self.reset()
            self.env.state = self.states[-1]
            
            if self.user_side == 1:
                self.model2_score += 1
            else:
                self.model1_score += 1

            if self.isGameEnd():
                self.model1_score = 0
                self.model2_score = 0

        return self.states, reward, done, launch
    
    def shuttleNetDyMF_shot(self, launch):
        done = False
        action, action_prob = self.model.action(self.states, self.actions)
        state, reward = self.env.step(action, launch)
        if reward != -1:
            self.states.append(state)
        if action is not None:
            self.actions.append(action)
            self.actions_prob.append(action_prob)

        launch = False

        # round end
        if reward == -1:
            launch = True
            self.user_launch = True
            done = True
            self.all_action.append(self.rally_actions)
            self.rally_actions = []
            turn_ = self.user_side

            # output data to dataFrame
            assert len(self.states) == len(self.actions) and  len(self.actions) == len(self.actions_prob)
            for i, (state, action, action_prob) in enumerate(zip(self.states, self.actions, self.actions_prob)):
                if i == len(self.states) - 1:
                    self.dumpData(turn_, state, action, action_prob, -1, i == 0) # last rally of round reward is -1
                elif i == 0:
                    self.dumpData(turn_, state, action, action_prob, 0, True)
                else:
                    self.dumpData(turn_, state, action, action_prob, 0, False)
                turn_ = 2 if turn_ == 1 else 1

            # _, _ = self.reset()
            self.env.state = self.states[-1]
            # launcher = 2
            
            if self.user_side == 1:
                self.model1_score += 1
            else:
                self.model2_score += 1

            if self.isGameEnd():
                self.model1_score = 0
                self.model2_score = 0

        return self.states, reward, done, launch