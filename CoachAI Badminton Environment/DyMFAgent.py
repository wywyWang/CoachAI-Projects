from RLModel import RLModel
import pandas as pd
import numpy as np
import ast
import torch
import os
import random
from MovementForecasting.DyMF.model import Encoder, Decoder
from MovementForecasting.DyMF.runner import predictAgent
import pickle
from typing import Literal

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # gpu vars

def load_args_file(model_folder):
    with open(model_folder + '/config', 'rb') as f:
        args = pickle.load(f)
    return args

mid_point = [177.5, 480]

class DyMFAgent(RLModel):
    def __init__(self, player: Literal[1,2]):
        super().__init__()
        
        # at top field(1) or bottom field(2)
        self.position = player

        self.model_path = "./MovementForecasting/model/DyMF_2_20230905_150"
        self.args = load_args_file(self.model_path)#model_folder
        self.args['sample_num'] = self.args['max_length']#int(sample_num)

        np.random.seed(self.args['seed'])
        random.seed(self.args['seed'])
        torch.manual_seed(self.args['seed'])
        torch.cuda.manual_seed(self.args['seed'])
        torch.cuda.manual_seed_all(self.args['seed'])
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.args['uniques_type'] = ['發短球', '長球', '推撲球', '殺球', '接殺防守', '平球', '網前球', '挑球', '切球', '發長球']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def state2dataset(self, states, actions, players: list):
        states = states[-self.args['max_length']+1:]
        actions = actions[-self.args['max_length']+1:]
        player=np.ones(len(actions),dtype=int)
        shot_type=np.zeros(len(actions),dtype=int)
        player_A_x=np.zeros(len(actions),dtype=float)
        player_A_y=np.zeros(len(actions),dtype=float)
        player_B_x=np.zeros(len(actions),dtype=float)
        player_B_y=np.zeros(len(actions),dtype=float)
       
        for i in range(len(actions)):
            player_coord, opponent_coord, ball_coord = states[i+1]
            shot, landing_coord, move_coord = actions[i]

            # next ball is current player
            # so last ball(note as n ball) is opponent
            # we find that count from last ball
            # every even ball is current player
            if (len(actions) - i)%2 == 0:
                player[i] = players[1]
                playerA_coord = self.subj2obj_coord(player_coord, 3 - self.position)
                playerB_coord = self.subj2obj_coord(opponent_coord, 3 - self.position)
                #player_coord = self.discrete2continuous(player_region, 1)
                #opponent_coord = self.discrete2continuous(opponent_region, 2)
            else:
                player[i] = players[0]
                playerB_coord = self.subj2obj_coord(player_coord, self.position)
                playerA_coord = self.subj2obj_coord(opponent_coord, self.position)
                #opponent_coord = self.discrete2continuous(player_region, 2)
                #player_coord = self.discrete2continuous(opponent_region, 1)

            player_A_x[i] = playerA_coord[0]
            player_A_y[i] = playerA_coord[1]
            player_B_x[i] = playerB_coord[0]
            player_B_y[i] = playerB_coord[1]
            shot_type[i] = shot

        mean_x, std_x = 175., 82.
        mean_y, std_y = 467., 192.
        player_A_x = (player_A_x - mean_x)/std_x
        player_A_y = (player_A_y - mean_y)/std_y
        player_B_x = (player_B_x - mean_x)/std_x
        player_B_y = (player_B_y - mean_y)/std_y

        return player, shot_type, player_A_x, player_A_y, player_B_x, player_B_y

    # shuttleNet need last five state
    # state: (player, opponent, ball)
    def action(self, states, actions, players):
        datas = self.state2dataset(states, actions, players) 

        encoder = Encoder(self.args, self.device)
        decoder = Decoder(self.args, self.device)
        encoder.player_embedding.weight = decoder.player_embedding.weight
        encoder.coordination_transform.weight = decoder.coordination_transform.weight

        try:
            encoder.load_state_dict(torch.load(self.args['model_folder'] + '/encoder')), decoder.load_state_dict(torch.load(self.args['model_folder'] + '/decoder'))
        except:
            encoder.load_state_dict(torch.load(f'{self.model_path}/encoder')), decoder.load_state_dict(torch.load(f'{self.model_path}/decoder'))

        encoder.to(self.device), decoder.to(self.device)


        shot_type, A_x, A_y, B_x, B_y, gmm_param_A, gmm_param_B = predictAgent(datas, encoder, decoder, self.args, device=self.device)

        mean_x, std_x = 175., 82.
        mean_y, std_y = 467., 192.
        B_x = B_x * std_x + mean_x # player_A_x
        B_y = B_y * std_y + mean_y # player_A_x
        B_x, B_y = self.obj2subj_coord((B_x, B_y), self.position)

        return B_x.item(), B_y.item(), gmm_param_B


        

