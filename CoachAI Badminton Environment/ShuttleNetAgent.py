from RLModel import RLModel
import pandas as pd
import numpy as np
from StrokeForecasting.badmintoncleaner import prepare_testDataset
import ast
from StrokeForecasting.ShuttleNet.Decoder import ShotGenDecoder
from StrokeForecasting.ShuttleNet.Encoder import ShotGenEncoder
from StrokeForecasting.utils import predictAgent
import torch
from typing import Tuple, Sequence, Literal

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # gpu vars

mid_point = [177.5, 480]

class ShuttleNetAgent(RLModel):
    def __init__(self, player: Literal[1, 2]):
        super().__init__()

        self.position = player

        self.model_path = "./StrokeForecasting/evaluate_3GMM90"
    
        self.config = ast.literal_eval(open(f"{self.model_path}1/config", encoding='utf8').readline())
        set_seed(self.config['seed_value'])

        print(self.config['uniques_type'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.type_mapping = {'發短球': 1, '長球': 2, '推撲球': 3, '殺球':4, '接殺防守':5, '平球':6, '網前球':7, '挑球':8, '切球':9, '發長球':10, '接不到':11} 

        print(self.model_path)

    def state2dataset(self, states:Sequence[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]],
                      actions:Sequence[Tuple[int, Tuple[float, float], Tuple[float, float]]], players:Sequence[int]):
        shot_type=np.zeros(len(actions),dtype=int)
        landing_x=np.zeros(len(actions),dtype=float)
        landing_y=np.zeros(len(actions),dtype=float)
        #landing_region = np.zeros(5, dtype=int)
        player=np.ones(len(actions),dtype=int)
        for i in range(len(actions)):
            player_coord, opponent_coord, ball_coord = states[i]
            shot, landing_coord, move_coord = actions[i]

            # next ball is current player
            # so last ball(note as n ball) is opponent
            # we find that count from last ball
            # every even ball is current player
            if (len(actions) - i)%2 == 1:
                #landing_coord = self.discrete2continuous(landing_region, 1)
                player[i] = players[1]
                landing_x[i], landing_y[i] = self.subj2obj_coord(landing_coord, 3 - self.position)
            else:
                #landing_coord = self.discrete2continuous(landing_region, 2)
                player[i] = players[0]
                landing_x[i], landing_y[i] = self.subj2obj_coord(landing_coord, self.position)

            shot_type[i] = shot
            #landing_region[i] = ball_region

        mean_x, std_x = 175., 82.
        mean_y, std_y = 467., 192.
        landing_x = (landing_x - mean_x)/std_x
        landing_y = (landing_y - mean_y)/std_y

        return (shot_type, landing_x, landing_y, player)

    # shuttleNet need last five state
    # state: (player(x, y), opponent(x, y), ball(x, y))
    # action: (shot type, land(x, y), move(x, y))
    def action(self, states:Sequence[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]], 
               actions:Sequence[Tuple[int, Tuple[float, float], Tuple[float, float]]], players:Sequence[int])\
         -> Tuple[int, float, float, list, Tuple[list, list, list]]:
        states = states[-self.config['max_ball_round']+1:]
        actions = actions[-self.config['max_ball_round']+1:]
        #if len(states) % 2 == 1:
        #    states = states[1:]
        #    actions = actions[1:]
        #states = states[-2:]
        #actions = actions[-2:]

        testData = self.state2dataset(states, actions, players)

        encoder = ShotGenEncoder(self.config)
        decoder = ShotGenDecoder(self.config)

        encoder.to(self.device), decoder.to(self.device)
        current_model_path = f"{self.model_path}1/"
        encoder_path = f"{current_model_path}encoder"
        decoder_path = f"{current_model_path}decoder"
        encoder.load_state_dict(torch.load(encoder_path)), decoder.load_state_dict(
            torch.load(decoder_path))


        # run prediction
        result = predictAgent(testData, encoder, decoder, self.config, device=self.device)

        #type, x, y, sx, sy, corr = result
        type, landing_x, landing_y, shot_distribution, gmm_param = result

        type = self.type_mapping[self.config['uniques_type'][type]]

        mean_x, std_x = 175., 82.
        mean_y, std_y = 467., 192.
        landing_x = landing_x * std_x + mean_x
        landing_y = landing_y * std_y + mean_y
        landing_x, landing_y = self.obj2subj_coord((landing_x, landing_y), self.position)
        #return (type, self.coordcontinuous2discrete(x,y))
        return type, landing_x, landing_y, shot_distribution, gmm_param

