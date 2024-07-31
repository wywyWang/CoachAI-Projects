import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import pickle

from pickle import TRUE
from policy.preprocess.tool import *
from policy.preprocess.helper import *
from policy.transformer.encoder import Encoder
from copy import copy, deepcopy
from torch.nn.utils.rnn import pad_sequence
from dtaidistance import dtw_ndim
from torchsde.settings import LEVY_AREA_APPROXIMATIONS, METHODS, NOISE_TYPES, SDE_TYPES
from torchsde._brownian import BaseBrownian, BrownianInterval
from torchsde._core import misc


EVAL_LEN = None
GIVEN_FLAG = None
def write_mission(given_first2):
    global EVAL_LEN
    global GIVEN_FLAG
    if given_first2 == "True":
        print("given states of first two steps")
        EVAL_LEN = 18
        GIVEN_FLAG = True
    else:
        print("given initial state only")
        EVAL_LEN = 20
        GIVEN_FLAG = False

def final_sample_expert_eval(gen, config, num_samples, sample_from, no_touch, f_or_h):
    """for predicted action saving"""
    player_samples = torch.zeros(num_samples, gen.max_seq_len, gen.states_dim).type(torch.FloatTensor)
    opponent_samples = torch.zeros(num_samples, gen.max_seq_len, gen.states_dim).type(torch.FloatTensor)
    given_player_samples = torch.zeros(num_samples, gen.max_seq_len, gen.states_dim).type(torch.FloatTensor)
    given_opponent_samples = torch.zeros(num_samples, gen.max_seq_len, gen.states_dim).type(torch.FloatTensor)
    
    samples = torch.zeros(num_samples, gen.max_seq_len, gen.states_dim).type(torch.FloatTensor)
    sample_actions = torch.zeros(num_samples, gen.max_seq_len, gen.actions_dim).type(torch.FloatTensor)
    sample_expert_actions = torch.zeros(num_samples, gen.max_seq_len,gen.actions_dim).type(torch.FloatTensor)
    sample_actions_shot_logit = []

    vae_lmbd_latent_mean = []
    vae_lmbd_latent_logvar = []

    op_samples = torch.zeros(num_samples, gen.max_seq_len, gen.states_dim).type(torch.FloatTensor)
    op_sample_actions = torch.zeros(num_samples, gen.max_seq_len,gen.actions_dim).type(torch.FloatTensor)
    op_sample_expert_actions = torch.zeros(num_samples, gen.max_seq_len,gen.actions_dim).type(torch.FloatTensor)

    op_sample_actions_shot_logit = []

    op_vae_lmbd_latent_mean = []
    op_vae_lmbd_latent_logvar = []

    
    """for decoded action saving"""
    
    context_actions = torch.zeros(num_samples, gen.max_seq_len, gen.actions_dim).type(torch.FloatTensor)
    op_context_actions = torch.zeros(num_samples, gen.max_seq_len, gen.actions_dim).type(torch.FloatTensor)
    context_actions_shot_logit = []
    op_context_actions_shot_logit = []

    """sample data"""
    expert_states, expert_actions, op_expert_states, op_expert_actions, record = sample_inputs(num_samples, sample_from, f_or_h, no_touch)

    
    """battle setting"""
    eval_len = len(expert_states)
    op_eval_len = len(op_expert_states)
    stop_flag = [0] * eval_len
    op_stop_flag = [0] * op_eval_len
    ctx_stop_flag = [0] * eval_len
    op_ctx_stop_flag = [0] * op_eval_len
    samples = samples.cuda()
    first_flag = True        

    
    """brownian motion setting"""
    bm = BrownianInterval(t0=0, t1=2*gen.max_seq_len, size=(num_samples, gen.state_embed_dim+1),\
    dtype=samples.dtype, device=samples.device, levy_area_approximation=LEVY_AREA_APPROXIMATIONS.none,\
    entropy= config['seed'], tol=1e-5, halfway_tree=True)
    step_size = gen.step_size
    prev_t = curr_t = 0
    zs = []

    for i in range(gen.max_seq_len):
        for p in range(2):
            if p == 0:
                for j in range(eval_len):
                    if i < len(expert_actions[j]):
                        player_num = int(expert_states[j][i][17].item())
                        insert_expert_actions = expert_actions[j][i][:].cuda()
                        insert_sample = expert_states[j][i][:].cuda()
                        sample_expert_actions[j, i, :] = insert_expert_actions
                        samples[j, i, :] = insert_sample
                        if i == 0:
                            player_samples[j, i, :] = insert_sample
                        else:
                            inp_states = get_next_player_state(op_sample_actions[j, i-1, :].data, opponent_samples[j, i-1, :].data, player_num)
                            inp_states = inp_states.cuda()
                            player_samples[j, i, :] = inp_states.view(1,gen.states_dim).data
                            if f_or_h ==  "h":
                                if player_samples[j, i, 0] == 0 :
                                    stop_flag[j] = 1
                        if i == 0 or i == 1:
                            given_player_samples[j, i, :] = insert_sample
                    else:
                        if f_or_h == "f":
                            stop_flag[j] = 1
                        else:
                            inp_states = get_next_player_state(op_sample_actions[j, i-1, :].data, opponent_samples[j, i-1, :].data, player_num)
                            inp_states = inp_states.cuda()
                            player_samples[j, i, :] = inp_states.view(1,gen.states_dim).data
                            if player_samples[j, i, 0] == 0 :
                                stop_flag[j] = 1
            else:
                for j in range(op_eval_len):
                    if i < len(op_expert_actions[j]):
                        opponent_num = int(op_expert_states[j][i][17].item())
                        op_insert_expert_actions = op_expert_actions[j][i][:].cuda()
                        op_insert_sample = op_expert_states[j][i][:].cuda()
                        op_sample_expert_actions[j, i, :] = op_insert_expert_actions
                        op_samples[j, i, :] = op_insert_sample

                        inp_states = get_next_player_state(sample_actions[j, i, :].data, player_samples[j, i, :].data, opponent_num)
                        inp_states = inp_states.cuda()
                        opponent_samples[j, i, :] = inp_states.view(1,gen.states_dim).data
                        if f_or_h == "h":
                            if opponent_samples[j, i, 0] == 0 :
                                op_stop_flag[j] = 1
                        if i == 0 or i == 1:
                            given_opponent_samples[j, i, :] = op_insert_sample
                    else:
                        if f_or_h == "f":
                            op_stop_flag[j] = 1
                        else:
                            inp_states = get_next_player_state(sample_actions[j, i, :].data, player_samples[j, i, :].data, opponent_num)
                            inp_states = inp_states.cuda()
                            opponent_samples[j, i, :] = inp_states.view(1,gen.states_dim).data
                            if opponent_samples[j, i, 0] == 0 :
                                op_stop_flag[j] = 1
            
            """action projection layer"""

            if p == 0:
                if f_or_h == "f":
                    z_f, f, z_h, h = gen.forward(player_samples, expert_actions.copy(), player_samples[:, i, :], f_or_h)
                else:
                    if (i == 0 or i == 1) and GIVEN_FLAG == True:
                        z_h, h = gen.forward(player_samples, expert_actions.copy(), given_player_samples[:, i, :], f_or_h)
                    else:
                        z_h, h = gen.forward(player_samples, expert_actions.copy(), player_samples[:, i, :], f_or_h)

                vae_lmbd_latent_mean.append(gen.subspace_lmbd.latent_mean)
                vae_lmbd_latent_logvar.append(gen.subspace_lmbd.latent_logvar)
                
                # context decoder
                context_land_logit = gen.context_land_area(gen._ctx)
                context_shot_logit = gen.context_shot_type(gen._ctx)
                context_move_logit = gen.context_move_area(gen._ctx)
                context_actions_shot_logit.append(context_shot_logit)
            else:
                if f_or_h == "f":
                    z_f, f, z_h, h = gen.forward(opponent_samples, op_expert_actions.copy(), opponent_samples[:, i, :], f_or_h)
                else:
                    if i == 0 and GIVEN_FLAG == True:
                        z_h, h = gen.forward(opponent_samples, op_expert_actions.copy(), given_opponent_samples[:, i, :], f_or_h)
                    z_h, h = gen.forward(opponent_samples, op_expert_actions.copy(), opponent_samples[:, i, :], f_or_h)
                
                op_vae_lmbd_latent_mean.append(gen.subspace_lmbd.latent_mean)
                op_vae_lmbd_latent_logvar.append(gen.subspace_lmbd.latent_logvar)
                
                # context decoder                 
                context_land_logit = gen.context_land_area(gen._ctx)
                context_shot_logit = gen.context_shot_type(gen._ctx)
                context_move_logit = gen.context_move_area(gen._ctx)
                op_context_actions_shot_logit.append(context_shot_logit)


            """SDE integrate"""

            if first_flag == True:
                global z0
                if f_or_h == "f":
                    z0 = torch.cat((z_f, z_f.new_zeros(size=(z_f.size(0), 1))), dim=1)
                else:
                    z0 = torch.cat((z_h, z_h.new_zeros(size=(z_h.size(0), 1))), dim=1)
                global prev_z
                global curr_z 
                prev_z = curr_z = z0
                zs.append(z0)
                first_flag = False
                next_t = curr_t + step_size
                I_k = bm(curr_t, next_t)
            else:
                next_t = curr_t + step_size
                prev_t, prev_z = curr_t, curr_z
                I_k = bm(curr_t, next_t)
            
            # f_and_g_diagonal
            if f_or_h == "f":
                g = gen.g(z_f)
            else:
                g = gen.g(z_h)
            
            if f_or_h == "f":
                u = misc.stable_division(f - h, g)
                f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
                g_logqp = z_f.new_zeros(size=(z_f.size(0), 1))
            else:
                g_logqp = z_h.new_zeros(size=(z_h.size(0), 1))

            if f_or_h == "f":
                f = torch.cat([f, f_logqp], dim=1)
            else:
                f = torch.cat([h, g_logqp], dim=1)
            
            g = torch.cat([g, g_logqp], dim=1)
            g_prod = g * I_k
            dt = next_t - curr_t
            curr_z = curr_z + f * dt + g_prod
            curr_t = next_t
            zs.append(curr_z)
            project_embedding, _ = curr_z.split(split_size=(z0.size(1) - 1, 1), dim=1)

            '''
            sample predicted action
            '''
            land_logit = gen.predict_land_area(project_embedding)
            shot_logit = gen.predict_shot_type(project_embedding)
            move_logit = gen.predict_move_area(project_embedding)
            
            land_logit[land_logit==0] = eps
            move_logit[move_logit==0] = eps

            land_gmm = D.MixtureSameFamily(D.Categorical(gen.softmax(land_logit[:,:gen.gmm_num])),\
            D.Independent(D.Normal(torch.reshape(land_logit[:,gen.gmm_num:gen.gmm_num*3],(-1,gen.gmm_num,2)),\
            torch.reshape(torch.abs(torch.clamp(land_logit[:,gen.gmm_num*3:],-gen.gmm_thres,gen.gmm_thres)),(-1,gen.gmm_num,2))), 1))

            move_gmm = D.MixtureSameFamily(D.Categorical(gen.softmax(move_logit[:,:gen.gmm_num])),\
            D.Independent(D.Normal(torch.reshape(move_logit[:,gen.gmm_num:gen.gmm_num*3],(-1,gen.gmm_num,2)),\
            torch.reshape(torch.abs(torch.clamp(move_logit[:,gen.gmm_num*3:],-gen.gmm_thres,gen.gmm_thres)),(-1,gen.gmm_num,2))), 1))

            agent_shot_type = torch.multinomial(torch.exp(gen.log_softmax(shot_logit)), 1)
            while agent_shot_type[0, 0] == 0:
                agent_shot_type = torch.multinomial(torch.exp(gen.log_softmax(shot_logit)), 1)
            
            acts = torch.cat((land_gmm.sample(), agent_shot_type, move_gmm.sample()),1)

            """
            sample decoded action
            """
            context_land_logit[context_land_logit==0] = eps
            context_move_logit[context_move_logit==0] = eps

            context_land_gmm = D.MixtureSameFamily(D.Categorical(gen.softmax(context_land_logit[:,:gen.gmm_num])),\
            D.Independent(D.Normal(torch.reshape(context_land_logit[:,gen.gmm_num:gen.gmm_num*3],(-1,gen.gmm_num,2)),\
            torch.reshape(torch.abs(torch.clamp(context_land_logit[:,gen.gmm_num*3:],-gen.gmm_thres,gen.gmm_thres)),(-1,gen.gmm_num,2))), 1))
            
            context_move_gmm = D.MixtureSameFamily(D.Categorical(gen.softmax(context_move_logit[:,:gen.gmm_num])),\
            D.Independent(D.Normal(torch.reshape(context_move_logit[:,gen.gmm_num:gen.gmm_num*3],(-1,gen.gmm_num,2)),\
            torch.reshape(torch.abs(torch.clamp(context_move_logit[:,gen.gmm_num*3:],-gen.gmm_thres,gen.gmm_thres)),(-1,gen.gmm_num,2))), 1))

            context_shot_type = torch.multinomial(torch.exp(gen.log_softmax(context_shot_logit)), 1)
            while context_shot_type[0, 0] == 0:
                context_shot_type = torch.multinomial(torch.exp(gen.log_softmax(context_shot_logit)), 1)
            
            context_acts = torch.cat((context_land_gmm.sample(), context_shot_type, context_move_gmm.sample()),1)

            if p == 0:
                sample_actions[:, i, :] = acts.view(num_samples,gen.actions_dim).data
                context_actions[:, i, :] = context_acts.view(num_samples,gen.actions_dim).data
                sample_actions_shot_logit.append(torch.exp(gen.log_softmax(shot_logit)))

                for j in range(eval_len):
                    if stop_flag[j] == 1:
                        sample_actions[j, i, :] = torch.zeros(gen.actions_dim).type(torch.FloatTensor)

                        sample_actions_shot_logit[-1] = torch.zeros_like(sample_actions_shot_logit[-1])
                        sample_actions_shot_logit[-1][0][0] = torch.FloatTensor([1])
                        sample_actions_shot_logit[-1][0][1:] = torch.FloatTensor([eps])
                    
                    if check_tatic_next_status(context_actions[j, i, :].data):
                        ctx_stop_flag[j] = 1
                    
                    if ctx_stop_flag[j] == 1:
                        context_actions[j, i, :] = torch.zeros(gen.actions_dim).type(torch.FloatTensor)
                stop_flag = [0] * eval_len
            else:
                op_sample_actions[:, i, :] = acts.view(num_samples,gen.actions_dim).data
                op_context_actions[:, i, :] = context_acts.view(num_samples,gen.actions_dim).data
                op_sample_actions_shot_logit.append(torch.exp(gen.log_softmax(shot_logit)))

                for j in range(op_eval_len):
                    if op_stop_flag[j] == 1:
                        op_sample_actions[j, i, :] = torch.zeros(gen.actions_dim).type(torch.FloatTensor)

                        op_sample_actions_shot_logit[-1] = torch.zeros_like(op_sample_actions_shot_logit[-1])
                        op_sample_actions_shot_logit[-1][0][0] = torch.FloatTensor([1])
                        op_sample_actions_shot_logit[-1][0][1:] = torch.FloatTensor([eps])
                    
                    if check_tatic_next_status(op_context_actions[j, i, :].data):
                        op_ctx_stop_flag[j] = 1
                    
                    if op_ctx_stop_flag[j] == 1:
                        op_context_actions[j, i, :] = torch.zeros(gen.actions_dim).type(torch.FloatTensor)
                op_stop_flag = [0] * op_eval_len
    
    """re-choose context"""
    gen._ctx = None

    if GIVEN_FLAG == True:
        sample_actions[:,:2,:] = sample_expert_actions[:,:2,:]
        op_sample_actions[:,:1,:] = op_sample_expert_actions[:,:1,:]
    
    SDEs_KLD = 0
    return sample_actions, sample_expert_actions,\
    sample_actions_shot_logit,\
    op_sample_actions, op_sample_expert_actions,\
    op_sample_actions_shot_logit,\
    record, vae_lmbd_latent_mean, vae_lmbd_latent_logvar, op_vae_lmbd_latent_mean, op_vae_lmbd_latent_logvar, SDEs_KLD,\
    context_actions, context_actions_shot_logit,\
    op_context_actions, op_context_actions_shot_logit

def final_batchwise_sample(gen, config, num_samples, batch_size, sfrom, no_touch, f_or_h):
    
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
        final_sample_expert_eval(gen, config, batch_size, sfrom, no_touch, f_or_h)
        
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

def final_evaluate(gen, config, num_samples, batch_size, max_seq_len, sfrom, f_or_h):
    
    total_shot_ctc = 0
    total_land_dtw = 0
    total_move_dtw = 0

    op_total_shot_ctc = 0
    op_total_land_dtw = 0
    op_total_move_dtw = 0

    no_touch = []
    tnum_samples = num_samples
    for _ in trange(int(ceil(tnum_samples/int(batch_size)))):
        a, e, shot, oa, oe, oshot, no_touch, _, _, _, _, _,_, _, _, _ = final_batchwise_sample(gen, config, batch_size, batch_size, sfrom, no_touch, f_or_h)
        
        inp, target= a[:batch_size].cuda(), e[:batch_size].cuda()
        op_inp, op_target = oa[:batch_size].cuda(), oe[:batch_size].cuda()
    
        shot_ctc, land_dtw, move_dtw, len_list, gen_list, ans_list = final_batchNLLLoss(inp, target, shot[0], True)
        op_shot_ctc, op_land_dtw, op_move_dtw, op_len_list, op_gen_list, op_ans_list = final_batchNLLLoss(op_inp, op_target, oshot[0], False)

        skip_flag = 0
        if len_list[0] == 0:
            skip_flag += 1
        if op_len_list[0] == 0:
            skip_flag += 2
        
        if skip_flag == 0:
            pass
        elif skip_flag == 1:
            shot_ctc = op_shot_ctc
            land_dtw = op_land_dtw
            move_dtw = op_move_dtw
            len_list = op_len_list.copy() 
            gen_list = op_gen_list.copy()
            ans_list = op_ans_list.copy()
        elif skip_flag == 2:
            op_shot_ctc = shot_ctc
            op_land_dtw = land_dtw
            op_move_dtw = move_dtw
            op_len_list = len_list.copy()
            op_gen_list = gen_list.copy() 
            op_ans_list = ans_list.copy()
        elif skip_flag == 3:
            num_samples -= 1
            continue
        
        total_shot_ctc += shot_ctc / sum(len_list)
        total_land_dtw += land_dtw / sum(len_list)
        total_move_dtw += move_dtw / sum(len_list)

        op_total_shot_ctc += op_shot_ctc / sum(op_len_list)
        op_total_land_dtw += op_land_dtw / sum(op_len_list)
        op_total_move_dtw += op_move_dtw / sum(op_len_list)


    divider = (num_samples/batch_size)*2
    avg_shot_ce = total_shot_ctc/divider + op_total_shot_ctc/divider
    avg_land_dtw = total_land_dtw/divider + op_total_land_dtw/divider
    avg_move_dtw = total_move_dtw/divider + op_total_move_dtw/divider

    return avg_shot_ce, avg_land_dtw, avg_move_dtw

def final_batchNLLLoss(inp, target, shot_d, isplayer):
        
    if GIVEN_FLAG == True:
        if isplayer == True:
            start_index = 2
            inp = inp[:,start_index:,:]
            target = target[:,start_index:,:]
            shot_d = shot_d[start_index:]
        elif isplayer== False:
            start_index = 1
            inp = inp[:,start_index:,:]
            target = target[:,start_index:,:]
            shot_d = shot_d[start_index:]
    
    batch_size, seq_len = inp.size()[:2]

    shot_ctc = 0
    land_dtw = 0
    move_dtw = 0

    
    stp_list = [0]*batch_size
    len_list = [0]*batch_size
    ans_list = [0]*batch_size
    gen_list = [0]*batch_size
    
    flag_len_list = [0]*batch_size
    flag_gen_list = [0]*batch_size
    flag_ans_list = [0]*batch_size

    for j in range(batch_size):
        for i in range(seq_len):
            if (i < seq_len-1):
                if ((any(target[j, i, :].tolist()) == False) or (any(inp[j, i, :].tolist()) == False)):
                    if flag_len_list[j] == 0:
                        len_list[j] = i
                        stp_list[j] = 1
                        flag_len_list[j] = 1
                if (any(target[j, i, :].tolist()) == False):
                    if flag_ans_list[j] == 0:
                        ans_list[j] = i
                        flag_ans_list[j] = 1
                if (any(inp[j, i, :].tolist()) == False):
                    if flag_gen_list[j] == 0:
                        gen_list[j] = i
                        flag_gen_list[j] = 1

        land_dtw += dtw_ndim.distance(inp.data[j, :, :2].detach().cpu().numpy(), target.data[j, :, :2].cpu().numpy())
        move_dtw += dtw_ndim.distance(inp.data[j, :, 3:].detach().cpu().numpy(), target.data[j, :, 3:].cpu().numpy())
        shot_d = torch.stack(shot_d)
        targets = target.data[j, :, 2].type(torch.LongTensor)[:ans_list[0]]
        if F.ctc_loss(torch.log(shot_d.detach().cpu()), targets, torch.LongTensor([EVAL_LEN]), torch.LongTensor(ans_list), reduction = 'sum', zero_infinity = False).tolist() != np.inf:
            shot_ctc += F.ctc_loss(torch.log(shot_d.detach().cpu()), targets, torch.LongTensor([EVAL_LEN]), torch.LongTensor(ans_list), reduction = 'sum', zero_infinity = False)
        else:
            print("shot type evaluation error")
    return shot_ctc, land_dtw, move_dtw, len_list, gen_list, ans_list