import os
import gc
import sys
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

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--output_folder_name", type=str, help="path to save model")
parser.add_argument('--player_train', type=str, help="name of the player's training rallies")
parser.add_argument('--player_test', type=str, help="name of the player's testing rallies")
parser.add_argument('--opponent_train', type=str, help="name of the opponent's training rallies")
parser.add_argument('--opponent_test', type=str, help="name of the opponent's testing rallies")

parser.add_argument('--seed', type=str, default=46, help="seed value")
parser.add_argument('--cuda_position', type=int, default=0, help="position of GPU")
parser.add_argument("--max_ball_round", type=int, default=20, help="max of ball round")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--lr", type=int, default=1e-4, help="learning rate")
parser.add_argument("--dropout", type=int, default=0.1, help="proportion of dropout")
parser.add_argument("--epochs", type=int, default=30, help="epochs")
parser.add_argument("--K", type=int, default=0, help="number of fold for dataset")
parser.add_argument("--sample", type=int, default=300, help="number of samples for evaluation")
parser.add_argument("--experience_num", type=int, default=5, help="number of experience")
parser.add_argument("--context_dim", type=int, default=128, help="dimension of context")
parser.add_argument("--shot_num", type=int, default=12, help="number of shot type")
parser.add_argument("--shot_dim", type=int, default=32, help="dimension of shot type")
parser.add_argument("--player_dim", type=int, default=32, help="dimension of player")
parser.add_argument("--player_num", type=int, default=31, help="number of player")
parser.add_argument("--hidden_dim", type=int, default=256, help="dimension of hidden")
parser.add_argument("--actions_dim", type=int, default=5, help="dimension of actions")
parser.add_argument("--states_dim", type=int, default=18, help="dimension of states (with player_num)")
parser.add_argument("--GMM_num", type=int, default=5, help="number of bivariant Gaussian distributions")
parser.add_argument("--reg_weight", type=int, default=0.05, help="weight of regularization loss")
parser.add_argument("--GMM_thres", type=int, default=0.1, help="standard deviation threshold of bivariant Gaussian distributions")
config = vars(parser.parse_args())
args = parser.parse_args()

tool_global(args.K, args.experience_num, args.states_dim, args.actions_dim, [args.opponent_train, args.opponent_test,args.player_train])
saving_global(args.K, args.output_folder_name)

def set_seed(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed(args.seed)
torch.cuda.set_device(args.cuda_position)

output_folder_path = './Results/saved_model/{}/'.format(str(args.output_folder_name))
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

result_path = "Results/training_result/" + str(args.K) + "/" + str(args.output_folder_name) + ".txt"
try:
    os.remove(result_path)
except OSError as e:
    print(e)
else:
    print("File is deleted successfully")

def train_generator(epoch, gen, gen_opt):
    BATCH_SIZE = args.batch_size
    
    with open(result_path, 'a') as f:
        f.write("\n start train : ")
        f.write(str(datetime.now()))
        f.write("\n")
    
    for _ in trange(int(ceil(TRAIN_SAMPLES/int(BATCH_SIZE)))):
        no_touch = []
        a, e, shot, oa, oe, oshot, no_touch, mean, logvar, op_mean, op_logvar, SDEs_KLD,\
        ca, cshot, oca, ocshot = batchwise_sample(gen, config, BATCH_SIZE, BATCH_SIZE, expert_state_action, no_touch, "f")

        inp, target= a[:BATCH_SIZE].cuda(), e[:BATCH_SIZE].cuda()
        op_inp, op_target = oa[:BATCH_SIZE].cuda(), oe[:BATCH_SIZE].cuda()
        cinp = ca[:BATCH_SIZE].cuda()
        op_cinp = oca[:BATCH_SIZE].cuda()

        loss, len_list = gen.predict_batchNLLLoss(inp, target, shot[0], mean[0], logvar[0])
        op_loss, op_len_list = gen.predict_batchNLLLoss(op_inp, op_target, oshot[0], op_mean[0], op_logvar[0])

        decoder_loss = gen.decode_batchNLLLoss(cinp, target, cshot[0])
        op_decoder_loss = gen.decode_batchNLLLoss(op_cinp, op_target, ocshot[0])

        p_loss = ((SDEs_KLD + loss + op_loss)/(sum(len_list) + sum(op_len_list)))
        d_loss = ((decoder_loss + op_decoder_loss)/(sum(len_list) + sum(op_len_list)))

        gen_opt.zero_grad()
        (p_loss + d_loss).backward()
        gen_opt.step()

        del SDEs_KLD,inp,target,op_inp,op_target,\
        shot,mean,logvar,oshot,op_mean,op_logvar,\
        loss,op_loss, len_list, op_len_list,p_loss,d_loss,decoder_loss,op_decoder_loss,\
        cinp,op_cinp,cshot,ocshot

    shot_ce, land_dtw, land_mae, move_dtw, move_mae, JSD, MLD =\
    sample_evaluate(gen, config, TEST_SAMPLES, 1, args.max_ball_round, testing_state_action, "h")

    with open(result_path, 'a') as f:
        f.write('\n EPOCHS = %.4f' % epoch)
        f.write('\n predicted landing pos dtw = %.4f' % land_dtw)
        f.write('\n predicted moving pos dtw loss = %.4f' % move_dtw)
        f.write('\n predicted JSD = %.4f' % JSD)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable':trainable_num}

def main():
    
    training_rallies = pickle.load(open('Data/'+ str(args.K) + '/'+ str(args.player_train), 'rb'))
    testing_rallies = pickle.load(open('Data/'+ str(args.K) + '/'+ str(args.player_test), 'rb'))
    
    testing_rallies = testing_rallies[:args.sample]
    train_states, train_actions = state_action_separation(training_rallies)
    test_states, test_actions = state_action_separation(testing_rallies)

    no_state = [0]*(args.states_dim-1)
    no_action = [0]*args.actions_dim
    global expert_state_action
    expert_state_action = []
    for traj in training_rallies:
        if (len(traj) == 1) and (traj[:args.states_dim-1] == no_state) and (traj[-args.actions_dim:] == no_action):
            pass
        else:
            expert_state_action.append(traj)

    global testing_state_action
    testing_state_action = []
    for traj in testing_rallies:
        if (len(traj) == 1) and (traj[:args.states_dim-1] == no_state) and (traj[-args.actions_dim:] == no_action):
            pass
        else:
            testing_state_action.append(traj)

    global TRAIN_SAMPLES 
    TRAIN_SAMPLES = len(train_states)

    global TEST_SAMPLES 
    TEST_SAMPLES = len(test_states)

    gen = RallyNet(config)
    gen = gen.cuda()

    gen_optimizer = optim.Adam(gen.parameters(), lr=args.lr)    

    print('\nStarting Training...')
    print(get_parameter_number(gen))
    gen.train()

    for epoch in range(args.epochs):
        print('--------\nEPOCH %d\n--------' % (epoch+1))
        print('Training Generator : \n', end='')
        sys.stdout.flush()
        gc.collect()
        train_generator(epoch, gen, gen_optimizer)
        torch.save(gen.state_dict(), output_folder_path+'gen_e_{}.trc'.format(str(epoch)))

        with open(output_folder_path + "config", 'w') as config_file:
            config_file.write(str(config))

if __name__ == "__main__":
    main()
