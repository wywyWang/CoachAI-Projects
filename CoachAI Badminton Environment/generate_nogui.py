import pandas as pd
from typing import Tuple
import numpy as np
import pickle
import tqdm
from RLEnvironment import Env
from ppo import PPO
import torch
from ddpg import DDPG
from a2c import A2C


class GenerateThread:
    def __init__(self, output_filename: str):
        self.output_filename = output_filename
        self.shottype_mapping = ['發短球', '長球', '推球', '殺球',
                                 '擋小球', '平球', '放小球', '挑球', '切球', '發長球', '接不到']
        self.round = 1
        self.rally = 0
        self.output = pd.DataFrame()
        self.model1_score = 0
        self.model2_score = 0

    # convert action, stage based data to rally base
    def dumpData(self, player: int, state: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                 action: Tuple[int, Tuple[float, float], Tuple[float, float]],
                 action_prob: Tuple[list, Tuple[list, list, list], Tuple[list, list, list]], reward: int, is_launch: bool):
        # launch failed
        # if is_launch and reward == -1:
        #    return
        state_player, state_opponent, state_ball = state
        action_type, action_land, action_move = action
        action_type_prob, action_land_gmm_param, action_move_gmm_param = action_prob

        if is_launch:
            self.rally += 1

        # 11 mean cannot reach, prev state is last state
        # if action_type != 11:
        player_x, player_y = action_move
        landing_x, landing_y = action_land
        row = pd.DataFrame([{'rally': self.rally,
                            'obs_ball_round': self.round,
                             'obs_player': 'A' if player == 1 else 'B',
                             'A_score': self.model1_score,
                             'B_score': self.model2_score,
                             'obs_serve': is_launch,
                             'act_landing_location_x': landing_x,
                             'act_landing_location_y': landing_y,
                             'act_player_defend_x': player_x,
                             'act_player_defend_y': player_y,
                             'shot_prob': action_type_prob,
                             'land_prob': action_land_gmm_param,
                             'move_prob': action_move_gmm_param,
                             'act_ball_type': self.shottype_mapping[int(action_type)-1]}])
        self.output = pd.concat([self.output, row])

        self.round += 1

        # current rally end
        if reward == -1:
            self.round = 1

    def save(self):
        self.output.to_csv(self.output_filename, index=False)


class GenerateThread2Agent(GenerateThread):
    def __init__(self, rally_count: int, model1_path: str, is_model1_shuttleNet: bool, model1_shuttleNet_player: int,
                 model2_path: str, is_model2_shuttleNet: bool, model2_shuttleNet_player: int,
                 output_filename: str):
        super().__init__(output_filename)
        self.output_filename = output_filename
        self.model1_path = model1_path
        self.model2_path = model2_path
        self.is_model1_shuttleNet = is_model1_shuttleNet
        self.is_model2_shuttleNet = is_model2_shuttleNet
        self.cnt = 0
        self.model1_shuttleNet_player = model1_shuttleNet_player
        self.model2_shuttleNet_player = model2_shuttleNet_player
        self.all_action = []
        self.rally_actions = []
        self.rally_count = rally_count

        # get first n ball from history data
        # needed ball round + 1 and filtered out last row so that last ball is not end
        if self.is_model1_shuttleNet or self.is_model2_shuttleNet:
            print('contain ShuttleNet')
            self.init_row_count = 2  # for shuttleNet
        else:
            self.init_row_count = 2
        data = pd.read_csv('StrokeForecasting/data/continous_subjective.csv')
        data = data[['rally_id', 'type', 'rally', 'ball_round', 'landing_x', 'landing_y',
                     'player_location_x', 'player_location_y', 'opponent_location_x', 'opponent_location_y']]
        grouped = data.groupby(['rally_id'])
        filtered = grouped.filter(lambda x: len(x) >= self.init_row_count)
        data.dropna(inplace=True)
        self.history_data = filtered.groupby(
            ['rally_id']).head(self.init_row_count)
        self.group_keys = list(
            self.history_data.groupby(['rally_id']).groups.keys())

        self.type_mapping = {'發短球': 1, '長球': 2, '推撲球': 3, '殺球': 4, '接殺防守': 5,
                             '平球': 6, '網前球': 7, '挑球': 8, '切球': 9, '發長球': 10, '接不到': 11}

    # sample start state from history data
    def sampleStartState(self):
        self.states = []

        self.actions = []
        self.actions_prob = []

        random_group_index = np.random.choice(len(self.group_keys))
        rows = self.history_data.groupby(['rally_id']).get_group(
            self.group_keys[random_group_index])

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
                self.actions_prob.append(([], [], []))

            prev_landing_coord = landing_coord
            prev_opponent_coord = opponent_coord
            prev_type = type

        # filtered last row to avoid data contain end ball
        self.states = self.states[:-1]
        self.actions = self.actions[:-1]
        self.actions_prob = self.actions_prob[:-1]

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

    def run(self):
        from SuperviseAgent import SuperviseAgent

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
        # print(self.states)
        # print(self.actions)
        # while not self.isGameEnd():
        launcher = 1
        is_launch = True
        for i in tqdm.tqdm(range(self.rally_count)):
            if turn == 1:
                if self.is_model1_shuttleNet:
                    action, action_prob = self.model1.action(
                        self.states, self.actions)
                else:
                    action, action_prob = self.model1.action(
                        self.states[-1], is_launch)
                # print(action_prob)
                # if action[0] == 11:
                #     tqdm.tqdm.write('cannot reach')
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
                    # print(len(self.states))
                    self.all_action.append(self.rally_actions)
                    self.rally_actions = []
                    turn_ = launcher

                    # output data to dataFrame
                    assert len(self.states) == len(self.actions) and len(
                        self.actions) == len(self.actions_prob)
                    for i, (state, action, action_prob) in enumerate(zip(self.states, self.actions, self.actions_prob)):
                        if i == len(self.states) - 1:
                            # last rally of round reward is -1
                            self.dumpData(turn_, state, action,
                                          action_prob, -1, i == 0)
                        elif i == 0:
                            self.dumpData(turn_, state, action,
                                          action_prob, 0, True)
                        else:
                            self.dumpData(turn_, state, action,
                                          action_prob, 0, False)
                        turn_ = 2 if turn_ == 1 else 1
                    self.sampleStartState()
                    self.env.state = self.states[-1]
                    launcher = 2

                    self.model2_score += 1

                    if self.isGameEnd():
                        self.outputScore()
                        self.model1_score = 0
                        self.model2_score = 0

                    # print(f'rally end, next launcher{launcher}')
            elif turn == 2:
                if self.is_model2_shuttleNet:
                    action, action_prob = self.model2.action(
                        self.states, self.actions)
                else:
                    action, action_prob = self.model2.action(
                        self.states[-1], is_launch)
                # print(action_prob)
                next_state, reward = self.env.step(action, is_launch)
                if reward != -1:
                    self.states.append(next_state)
                if action is not None:
                    self.actions.append(action)
                    # self.rally_actions.append(action)
                    self.actions_prob.append(action_prob)
                # if action[0] == 11:
                #     tqdm.tqdm.write('cannot reach')
                turn = 1
                is_launch = False

                # round end
                if reward == -1:
                    is_launch = True
                    # self.all_action.append(self.rally_actions)
                    # self.rally_actions = []
                    turn_ = launcher

                    assert len(self.states) == len(self.actions) and len(
                        self.actions) == len(self.actions_prob)
                    # output data to dataFrame
                    for i, (state, action, action_prob) in enumerate(zip(self.states, self.actions, self.actions_prob)):
                        if i == len(self.states) - 1:
                            self.dumpData(turn_, state, action,
                                          action_prob, -1, i == 0)
                        elif i == 0:
                            self.dumpData(turn_, state, action,
                                          action_prob, 0, True)
                        else:
                            self.dumpData(turn_, state, action,
                                          action_prob, 0, False)
                        turn_ = 2 if turn_ == 1 else 1

                    self.sampleStartState()
                    launcher = 1
                    self.env.state = self.states[-1]

                    self.model1_score += 1
                    if self.isGameEnd():
                        self.outputScore()
                        self.model1_score = 0
                        self.model2_score = 0
                    # print(f'rally end, next launcher{launcher}')

            # tqdm.tqdm.write(str(self.states[-1]))
        # np.save(f'./output/PPO_actions_1000000.npy', np.array(self.all_action, dtype=object))
        self.save()


if __name__ == '__main__':
    ### parameters ###
    max_rally_count = 10000
    is_model1_shuttleNet = True
    is_model2_shuttleNet = False
    model1_path = ''
    model2_path = './demo/PPO.pkl'
    ##################

    ################# PPO hyperparameters ################
    update_timestep = 1000 * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update
#
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
#
    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network
#
    random_seed = 0         # set random seed if required (0 = no random seed)
    ######################################################
#
    env = Env()
    # state space dimension
    state_dim = 6
    # action space dimension
    action_dim = 4
    # PPO
    print('PPO')
    check_step = 1_000_000  # 1_000_000
    output_filename = f'ppovsShuttleNet_{check_step}.csv'
    testing_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs,
                        eps_clip, has_continuous_action_space=True, train_step=max_rally_count/2)
    testing_agent.load(f'./ppo/data_train/PPO/PPO_{check_step}.pth')
#
    # for training
    # testing_agent.total_steps = check_step
    # testing_agent.start = check_step
    # testing_agent.isTrain = True
#
    testing_agent.save(model2_path)

    runner = GenerateThread2Agent(max_rally_count,
                                  model1_path, is_model1_shuttleNet, 2,
                                  model2_path, is_model2_shuttleNet, 2,
                                  output_filename)
    runner.run()

    # ################ DDPG hyperparameters ################
    # gamma = 0.99  # discount factor
    # tau = 0.005  # Softly update the target network
    # lr = 3e-4  # learning rate
    # #####################################################

    # DDPG
    # print('DDPG')
    # check_step = 100000
    # path_a = f'data_train/DDPG/DDPG_actor_{check_step}.pth'
    # path_c = f'data_train/DDPG/DDPG_critic_{check_step}.pth'
    # output_filename = f'output/ddpgvsShuttleNet_{check_step}.csv'
    # testing_agent = DDPG(state_dim, action_dim, gamma, tau, lr, max_rally_count/2)
    # testing_agent.load_(path_a, path_c)

    ################# A2C hyperparameters ################
    # gamma = 0.99  # discount factor
    # lr = 3e-4  # learning rate
    ######################################################

    # A2C
    # print('A2C')
    # check_step = 850_000
    # path_a = f'./a2c/data_train/A2C/A2C_actor_{check_step}.pth'
    # path_c = f'./a2c/data_train/A2C/A2C_critic_{check_step}.pth'
    # output_filename = f'a2cvsShuttleNet_{check_step}.csv'
    # testing_agent = A2C(state_dim, action_dim, gamma, lr, max_rally_count/2)
    # testing_agent.load_(path_a, path_c)
