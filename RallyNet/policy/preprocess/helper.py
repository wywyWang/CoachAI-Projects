import torch
from torch.autograd import Variable
from math import ceil
from policy.preprocess.tool import *
import numpy as np
import random
from math import log2
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm, trange

K = None
OUTPUT_NAME = None

def saving_global(mt, output_filename):
    global K
    global OUTPUT_NAME
    K = mt
    OUTPUT_NAME = output_filename

import sys
eps = sys.float_info.epsilon

def kl_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    p = p / np.sum(p)
    q = q / np.sum(q)

    p[p == 0] = eps
    q[q == 0] = eps
    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

def js_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    p = p / np.sum(p)
    q = q / np.sum(q)

    m = 0.5 * (p + q)

    p[p == 0] = eps
    q[q == 0] = eps
    m[m == 0] = eps
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def batchwise_sample(gen, config, num_samples, batch_size, sfrom, no_touch, f_or_h):
    
    """predict action"""
    sample_actions = []
    sample_expert_actions = []
    sample_shot_logit = []
    sample_mean = []
    sample_logvar = []

    op_sample_actions = []
    op_sample_expert_actions = []
    op_sample_shot_logit = []
    op_sample_mean = []
    op_sample_logvar = []
    
    """decode context"""
    
    context_actions_list = []
    context_shot_logit = []
    op_context_actions_list = []
    op_context_shot_logit = []

    for _ in range(int(ceil(int(num_samples)/int(batch_size)))):
        actions, expert_actions, actions_shot_logit,\
        op_actions, op_expert_actions, op_actions_shot_logit,\
        record, mean, logvar, op_mean, op_logvar, SDEs_KLD,\
        context_actions, context_actions_shot_logit,\
        op_context_actions, op_context_actions_shot_logit = \
        gen.sample_expert_eval(config, batch_size, sfrom, no_touch, f_or_h)
        
        no_touch.extend(record)

        """predict action"""
        sample_actions.append(actions)
        sample_expert_actions.append(expert_actions)
        sample_shot_logit.append(actions_shot_logit)
        sample_mean.append(mean)
        sample_logvar.append(logvar)

        op_sample_actions.append(op_actions)
        op_sample_expert_actions.append(op_expert_actions)
        op_sample_shot_logit.append(op_actions_shot_logit)
        op_sample_mean.append(op_mean)
        op_sample_logvar.append(op_logvar)

        """decode context"""
        context_actions_list.append(context_actions)
        context_shot_logit.append(context_actions_shot_logit)
        op_context_actions_list.append(op_context_actions)
        op_context_shot_logit.append(op_context_actions_shot_logit)

    """predict action"""
    return_sample_actions = torch.cat(sample_actions, 0)[:num_samples]
    return_sample_expert_actions = torch.cat(sample_expert_actions, 0)[:num_samples]
    
    op_return_sample_actions = torch.cat(op_sample_actions, 0)[:num_samples]
    op_return_sample_expert_actions = torch.cat(op_sample_expert_actions, 0)[:num_samples]
    
    """decode context"""
    return_context_actions = torch.cat(context_actions_list, 0)[:num_samples]
    op_return_context_actions = torch.cat(op_context_actions_list, 0)[:num_samples]

    return return_sample_actions, return_sample_expert_actions, sample_shot_logit,\
    op_return_sample_actions, op_return_sample_expert_actions, op_sample_shot_logit,\
    no_touch, sample_mean, sample_logvar, op_sample_mean, op_sample_logvar, SDEs_KLD,\
    return_context_actions, context_shot_logit,\
    op_return_context_actions, op_context_shot_logit

def sample_evaluate(gen, config, num_samples, batch_size, max_seq_len, sfrom, f_or_h):
    
    total_shot_ce = 0
    total_land_dtw = 0
    total_land_mae = 0
    total_move_dtw = 0
    total_move_mae = 0

    op_total_shot_ce = 0
    op_total_land_dtw = 0
    op_total_land_mae = 0
    op_total_move_dtw = 0
    op_total_move_mae = 0

    GLD = [0]*gen.max_seq_len
    ALD = [0]*gen.max_seq_len

    graphGenData = []
    graphAnsData = []
    
    no_touch = []
    tnum_samples = num_samples
    for _ in trange(int(ceil(tnum_samples/int(batch_size)))):
        a, e, shot, oa, oe, oshot, no_touch, _, _, _, _, _,_, _, _, _ = batchwise_sample(gen, config, batch_size, batch_size, sfrom, no_touch, f_or_h)
        
        inp, target= a[:batch_size].cuda(), e[:batch_size].cuda()
        op_inp, op_target = oa[:batch_size].cuda(), oe[:batch_size].cuda()
    
        shot_ce, land_mae, land_dtw, move_mae, move_dtw, len_list, gen_list, ans_list = gen.eval_batchNLLLoss(inp, target, shot[0])
        op_shot_ce, op_land_mae, op_land_dtw, op_move_mae, op_move_dtw, op_len_list, op_gen_list, op_ans_list = gen.eval_batchNLLLoss(op_inp, op_target, oshot[0])

        graphGenData.extend(gen_list)
        graphGenData.extend(op_gen_list)
        graphAnsData.extend(ans_list)
        graphAnsData.extend(op_ans_list)

        for length in gen_list:
            GLD[length] += 1
        for length in op_gen_list:
            GLD[length] += 1
        
        for length in ans_list:
            ALD[length] += 1
        for length in op_ans_list:
            ALD[length] += 1

        skip_flag = 0
        if len_list[0] == 0:
            skip_flag += 1
        if op_len_list[0] == 0:
            skip_flag += 2
        
        if skip_flag == 0:
            pass
        elif skip_flag == 1:
            shot_ce = op_shot_ce
            land_dtw = op_land_dtw
            land_mae = op_land_mae
            move_dtw = op_move_dtw
            move_mae = op_move_mae
            len_list = op_len_list.copy() 
            gen_list = op_gen_list.copy()
            ans_list = op_ans_list.copy()
        elif skip_flag == 2:
            op_shot_ce = shot_ce
            op_land_dtw = land_dtw
            op_land_mae = land_mae
            op_move_dtw = move_dtw
            op_move_mae = move_mae
            op_len_list = len_list.copy()
            op_gen_list = gen_list.copy() 
            op_ans_list = ans_list.copy()
        elif skip_flag == 3:
            num_samples -= 1
            continue
        
        total_shot_ce += shot_ce / sum(len_list)
        total_land_dtw += land_dtw / sum(len_list)
        total_land_mae += land_mae / sum(len_list)
        total_move_dtw += move_dtw / sum(len_list)
        total_move_mae += move_mae / sum(len_list)

        op_total_shot_ce += op_shot_ce / sum(op_len_list)
        op_total_land_dtw += op_land_dtw / sum(op_len_list)
        op_total_land_mae += op_land_mae / sum(op_len_list)
        op_total_move_dtw += op_move_dtw / sum(op_len_list)
        op_total_move_mae += op_move_mae / sum(op_len_list)


    divider = (num_samples/batch_size)*2

    avg_shot_ce = total_shot_ce/divider + op_total_shot_ce/divider
    avg_land_dtw = total_land_dtw/divider + op_total_land_dtw/divider
    avg_land_mae = total_land_mae/divider + op_total_land_mae/divider
    avg_move_dtw = total_move_dtw/divider + op_total_move_dtw/divider
    avg_move_mae = total_move_mae/divider + op_total_move_mae/divider
    
    JSD = js_divergence(GLD, ALD) 

    plt.cla() 
    sns.set()
    graphGenData = np.asarray(graphGenData)
    sns_plot = sns.kdeplot(graphGenData,fill=True, bw_adjust=1.5)
    graphAnsData = np.asarray(graphAnsData)
    sns_plot = sns.kdeplot(graphAnsData,fill=True, bw_adjust=1.5)
    plt.legend(labels=["generated length","correct length"])
    plt.savefig("Results/length_distribution/" + str(K) + "/" + OUTPUT_NAME + ".png")

    length_error = (np.asarray(graphAnsData) - np.asarray(graphGenData))
    MLD = np.sum(length_error) / len(graphAnsData)

    return avg_shot_ce, avg_land_dtw, avg_land_mae, avg_move_dtw, avg_move_mae, JSD, MLD
