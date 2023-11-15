from ShuttleNetAgent import ShuttleNetAgent
from DyMFAgent import DyMFAgent
from RLModel import RLModel
from typing import Sequence, Tuple, Literal
from First2Ball.BC import BCAgent
import torch
import torch.nn.functional as F

class SuperviseAgent(RLModel):
    def __init__(self, player: int, position: Literal[1, 2]):
        self.agent_first2ball = BCAgent()
        self.agent_first2ball.load('ShuttleDyMFNet2ball-final.pt')
        self.agent_shuttleNet = ShuttleNetAgent(position)
        self.agent_DyMFAgent = DyMFAgent(position)
        self.players = [0, player] #(custom agent, ShuttleNet agent)
        self.type_mapping = {'發短球': 1, '長球': 2, '推撲球': 3, '殺球':4, '接殺防守':5, '平球':6, '網前球':7, '挑球':8, '切球':9, '發長球':10, '接不到':11} 

    # need last 5 states and action to determine next action
    def action(self, states:Sequence[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]], 
               actions:Sequence[Tuple[int, Tuple[float, float], Tuple[float, float]]]):
        mean_x, std_x = 175., 82.
        mean_y, std_y = 467., 192.
        if len(states) <= 2:
            unique_type = ['發短球', '推撲球', '殺球', '網前球', '挑球', '發長球', '接殺防守', '切球', '平球', '長球', '接不到']
                          #  0         1           2    3       4       5         6          7       8       9         10
            is_launch = len(states) == 1
            #state = states[-1]
            player = torch.zeros(1,2,2)
            opponent = torch.zeros(1,2,2)
            ball = torch.zeros(1,2,2)
            for i in range(len(states)):
                player[0,0,i] = states[i][0][0]
                player[0,1,i] = states[i][0][1]
                opponent[0,0,i] = states[i][1][0]
                opponent[0,1,i] = states[i][1][1]
                ball[0,0,i] = states[i][2][0]
                ball[0,1,i] = states[i][2][1]

            player[:,:,0] /= std_x 
            opponent[:,:,0] /= std_x 
            ball[:,:,0] /= std_x 
            player[:,:,1] /= std_y 
            opponent[:,:,1] /= std_y 
            ball[:,:,1] /= std_y
            type = torch.zeros(1,2,dtype=torch.long)
            if len(states) == 2:
                type[0,1] = actions[0][0]

            predict = self.agent_first2ball.predict(player, opponent, ball, type)
            predict_shot, predict_land, predict_move = predict
            shot_prob = F.softmax(predict_shot, dim=-1)[:,1:] # remove padding
            type_prob = shot_prob[0].cpu().detach().numpy().tolist()
            while True:
                output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)
                type = output_shot[0, 0].item()
                if type in [0, 5] and is_launch: # launch
                    break
                if not type in [0, 5] and not is_launch: # launch
                    break
            type = self.type_mapping[unique_type[type]]

            landing_x = predict_land[0, 0].item() * std_x
            landing_y = predict_land[0, 1].item() * std_y
            player_region_x = predict_move[0, 0].item() * std_x
            player_region_y = predict_move[0, 1].item() * std_y
            landing_gmm_param = []
            player_region_gmm_param = []
            type_prob = [type_prob[0], type_prob[9], type_prob[1], type_prob[2], type_prob[6],
                         type_prob[8], type_prob[3], type_prob[4], type_prob[7], type_prob[5], type_prob[10]]
        else:
            player_region_x, player_region_y, player_region_gmm_param = self.agent_DyMFAgent.action(states, actions, self.players)
            type, landing_x, landing_y, type_prob, landing_gmm_param = self.agent_shuttleNet.action(states, actions, self.players)
            
            #print(type_prob, landing_region_prob)

        return (type, (landing_x, landing_y), (player_region_x, player_region_y)), \
               (type_prob, landing_gmm_param, player_region_gmm_param)