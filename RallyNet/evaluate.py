import os
import ast
import torch
import random
import pickle
import argparse
import torch.optim as optim

from io import open
from torch import nn as nn
from datetime import datetime
from policy.preprocess.tool import *
from policy.preprocess.helper import *
from policy.models.generator import *
from tqdm import tqdm, trange
from math import ceil

import utils.predict_func as predict_func

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def set_seed(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

model_name = sys.argv[1]
model_epoch = sys.argv[2] 
given_first2 = sys.argv[3]

model_path = 'Results/saved_model/' + str(model_name) + '/gen_e_' + str(model_epoch) + '.trc'
config = ast.literal_eval(open('Results/saved_model/' + str(model_name) + '/config').readline())

set_seed(config['seed'])
torch.cuda.set_device(config['cuda_position'])
cuda_location = "cuda:" + str(config['cuda_position'])

tool_global(config['K'], config['experience_num'], config['states_dim'], config['actions_dim'], [config['opponent_train'], config['opponent_test'], config['player_train']])
saving_global(config['K'], config['output_folder_name'])
predict_func.write_mission(given_first2)

testing_trajectories = pickle.load(open('Data/'+ str(config['K']) + '/' + config['player_test'], 'rb'))
test_states, test_actions = state_action_separation(testing_trajectories)

check_state_list = [0]*(config['states_dim']-1)
check_action_list = [0]*config['actions_dim']

global testing_state_action
testing_state_action = []
for traj in testing_trajectories:
    if (len(traj) == 1) and (traj[:config['states_dim']-1] == check_state_list) and (traj[-config['actions_dim']:] == check_action_list):
        pass
    else:
        testing_state_action.append(traj)

global TEST_SAMPLES 
TEST_SAMPLES = len(test_states)

def main():
    gen = RallyNet(config)
    gen.load_state_dict(torch.load(model_path, map_location = cuda_location))
    gen.to(torch.device(cuda_location))
    gen.eval()
    gen = gen.cuda()

    shot_ctc, land_dtw, move_dtw = predict_func.final_evaluate(gen, config, TEST_SAMPLES, 1, config['max_ball_round'], testing_state_action, "h")

    print('\n predicted shot type ctc = %.4f' % shot_ctc)
    print('\n predicted landing pos dtw = %.4f' % land_dtw)
    print('\n predicted moving pos dtw = %.4f' % move_dtw)

if __name__ == "__main__":
    main()
