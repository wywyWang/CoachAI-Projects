
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

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class Vae_Encoder(nn.Module):
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, block = 'LSTM'):

        super(Vae_Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length

        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout, batch_first=True)
        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout, batch_first=True)
        else:
            raise NotImplementedError

    def forward(self, x):
        _, (h_end, c_end) = self.model(x)
        h_end = h_end[-1, :, :]
        return h_end


class Vae_Lambda(nn.Module):
    def __init__(self, hidden_size, latent_length):
        super(Vae_Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean

class RallyNet(nn.Module):
    def __init__(self, config):
        super(RallyNet, self).__init__()
        self.states_dim = config['states_dim']
        self.max_seq_len = config['max_ball_round']
        self.actions_dim = config['actions_dim']
        self.context_dim = config['context_dim']
        self.shot_num = config['shot_num']
        self.log_softmax = nn.LogSoftmax(dim = -1)
        self.softmax = nn.Softmax(dim = -1)
        self.step_size = 1
        self._ctx = None

        self.encoder = Encoder(config['states_dim']-2, self.max_seq_len, 1, 1, config['dropout'])
        self.label_emb = nn.Embedding(config['player_num'], config['player_dim'])
        self.shot_emb = nn.Embedding(config['shot_num'], config['shot_dim'])

        self.state_embed_dim = config['states_dim']-2 + config['shot_dim'] + config['player_dim'] + config['context_dim']
        self.hidden_dim = config['hidden_dim']        
        self.alpha = config['reg_weight']
        self.gmm_num = config["GMM_num"]
        self.gmm_thres = config["GMM_thres"]
        
        self.state_ref_encoder = nn.Linear(self.state_embed_dim*self.max_seq_len, self.state_embed_dim, bias=False) # state + cxt

        self.f_decoder = nn.Sequential(
            nn.Linear(self.state_embed_dim, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.state_embed_dim),
        )

        self.predict_shifting = nn.Sequential(
            nn.Linear(self.state_embed_dim*self.max_seq_len, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.context_dim),
        )

        self.h_decoder = nn.Sequential(
            nn.Linear(self.state_embed_dim, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.state_embed_dim),
        )
        
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, self.state_embed_dim), # hidden size
                    nn.Softplus(),
                    nn.Linear(self.state_embed_dim, 1), # hidden size
                    nn.Sigmoid()
                )
                for _ in range(self.state_embed_dim) # latent size
            ]
        )

        self.predict_land_area = nn.Linear(self.state_embed_dim, self.gmm_num*5, bias=False) # "mux, muy, sx, sy, corr"
        self.predict_shot_type = nn.Linear(self.state_embed_dim, self.shot_num, bias=False)
        self.predict_move_area = nn.Linear(self.state_embed_dim, self.gmm_num*5,bias=False) # "mux, muy, sx, sy, corr"

        self.context_decoder = nn.Sequential(
            nn.Linear(self.context_dim, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.context_dim),
        )
        
        self.context_land_area = nn.Linear(self.context_dim, self.gmm_num*5, bias=False) # "mux, muy, sx, sy, corr"
        self.context_shot_type = nn.Linear(self.context_dim, self.shot_num,bias=False)
        self.context_move_area = nn.Linear(self.context_dim, self.gmm_num*5,bias=False) # "mux, muy, sx, sy, corr"

        # parameters for vae
        number_of_features = self.actions_dim
        latent_length = self.context_dim
        hidden_size = self.max_seq_len
        hidden_layer_depth = 5
        dropout_rate = 0.1
        block = 'LSTM'
        dtype = torch.FloatTensor

        self.subspace_encoder = Vae_Encoder(number_of_features = number_of_features,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               dropout=dropout_rate,
                               block=block)

        self.subspace_lmbd = Vae_Lambda(hidden_size=hidden_size,
                           latent_length=latent_length)

        self.init_params()
    
    
    def init_params(self):
        init_list = []
        pi = 0
        for name, param in self.named_parameters():
            pi += 1
            if ("g_nets" in name) or ("f_decoder" in name) or ("h_decoder" in name):
                init_list.append(pi)
            if ("predict_land_area" in name) or ("predict_shot_type" in name) or ("predict_move_area" in name):
                init_list.append(pi)

        init_index = 0
        for param in self.parameters():
            init_index += 1
            if param.dim() > 0 and (init_index in init_list):
                param.data.uniform_(-0.01, 0.01)
            param = param.cuda()


    def g(self, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, x, expert_actions, now_x_tensor, f_or_h, mask=None):
        player_condition = x[:,:,-1].detach().cpu().numpy().tolist()
        for batch_pc in player_condition:
            player_condition_batch = -1
            for player_index in range(len(batch_pc)):
                if player_index == 0:
                    player_condition_batch = batch_pc[player_index]
                else:
                    batch_pc[player_index] = player_condition_batch

        player_embedding = torch.reshape(self.label_emb(torch.from_numpy(np.asarray([player_condition])).type(torch.IntTensor).cuda()),(x.shape[0],x.shape[1],-1))
       

        x = x.cuda()
        player_embedding = player_embedding.cuda()

        x_no_shot = del_tensor_ele(x,12)
        e_outputs = self.encoder(x_no_shot[:,:,:-1], mask)
        shot_input = get_tensor_ele(x, 12)
        shot_input = shot_input.type(torch.IntTensor)
        shot_embedding = self.shot_emb(shot_input.cuda())
        e_outputs = torch.cat((e_outputs, shot_embedding), dim=2)

        state_embedding = torch.cat((e_outputs, player_embedding.cuda()), 2)

        """find reference and find experience
        According to the current state, collect rally with similar experience in the past,
        and use the action in the rally as the player's follow-up plan
        """
        now_x_list = now_x_tensor.view(-1, self.states_dim).tolist()
        now_area_loc = list(map(mapping_coord2area, now_x_list)) # [[player,ball,opponent], [player,ball,opponent], ...]
        intersection_lists = list(map(mapping_get_sim_traj_i, now_area_loc))        
        
        collection_sim_action_trajs_list = list(map(prepare_act_input,intersection_lists))
        collection_sim_action_trajs_list = list(map(prepare_enc_input, collection_sim_action_trajs_list))

        def contextualization(trajs_list):
            zero_padding = torch.zeros(self.max_seq_len,1)
            trajs_list = pad_sequence([trajs_list, zero_padding])[:,0,:]
            trajs_list = trajs_list.unsqueeze(0)            
            subspace_output = self.subspace_encoder(trajs_list.cuda())
            subspace_codes = self.subspace_lmbd(subspace_output)
            subspace_center = torch.mean(subspace_codes, dim=0, keepdim=True)
            return subspace_center

        """Contextualization is only needed for posterior inference. f(t,z)"""

        if f_or_h != "h":
            prior_z = list(map(contextualization, expert_actions))
            prior_z = torch.cat(prior_z)
            prior_z = prior_z.cuda()

            if self._ctx == None:
                self._ctx = self.context_decoder(prior_z)
            else:
                self._ctx = self.context_decoder(self._ctx)
            
            prior_z = torch.unsqueeze(prior_z, dim=1)
            prior_z = prior_z.repeat(1,self.max_seq_len,1)

            state_ref_emb = torch.cat((state_embedding, prior_z), 2)
            f_input = self.state_ref_encoder(state_ref_emb.view(x.size(0), -1))
            f_latent = self.f_decoder(f_input)
        else:
            prior_z = list(map(contextualization, expert_actions))
            prior_z = torch.cat(prior_z)
            prior_z = prior_z.cuda()

            if self._ctx == None:
                self._ctx = self.context_decoder(prior_z)
            else:
                self._ctx = self.context_decoder(self._ctx) 


        def cal_center(trajs_list):
            subspace_output = self.subspace_encoder(trajs_list.cuda())
            subspace_codes = self.subspace_lmbd(subspace_output)
            subspace_center = torch.mean(subspace_codes, dim=0, keepdim=True)
            return subspace_center
        
        latent_center = list(map(cal_center, collection_sim_action_trajs_list))
        latent_center = torch.cat(latent_center)
        latent_center = latent_center.cuda()

        latent_center_condition = torch.unsqueeze(latent_center, dim=1)
        latent_center_condition = latent_center_condition.repeat(1,self.max_seq_len,1)
        cpe_outputs = torch.cat((state_embedding, latent_center_condition), 2)

        predict_shifting_logit = self.predict_shifting(cpe_outputs.view(x.size(0), -1))
        return_style_latent_code = torch.add(latent_center,predict_shifting_logit)
        return_style_latent_code = torch.unsqueeze(return_style_latent_code, dim=1)
        return_style_latent_code = return_style_latent_code.repeat(1,self.max_seq_len,1)
        
        if f_or_h == "h":
            new_e_outputs = torch.cat((state_embedding, return_style_latent_code), 2)
        else:
            new_e_outputs = torch.cat((state_embedding, return_style_latent_code), 2)
        
        h_input = self.state_ref_encoder(new_e_outputs.view(x.size(0), -1))
        h_latent = self.h_decoder(h_input)

        if f_or_h == "h":
            return h_input, h_latent
        else:
            return f_input, f_latent, h_input, h_latent

    def sample_expert_eval(self, config, num_samples, sample_from, no_touch, f_or_h):
        
        """for predicted action saving"""
        
        player_samples = torch.zeros(num_samples, self.max_seq_len, self.states_dim).type(torch.FloatTensor)
        opponent_samples = torch.zeros(num_samples, self.max_seq_len, self.states_dim).type(torch.FloatTensor)
        
        samples = torch.zeros(num_samples, self.max_seq_len, self.states_dim).type(torch.FloatTensor)
        sample_actions = torch.zeros(num_samples, self.max_seq_len, self.actions_dim).type(torch.FloatTensor)
        sample_expert_actions = torch.zeros(num_samples, self.max_seq_len,self.actions_dim).type(torch.FloatTensor)
        sample_actions_shot_logit = []

        vae_lmbd_latent_mean = []
        vae_lmbd_latent_logvar = []

        op_samples = torch.zeros(num_samples, self.max_seq_len, self.states_dim).type(torch.FloatTensor)
        op_sample_actions = torch.zeros(num_samples, self.max_seq_len,self.actions_dim).type(torch.FloatTensor)
        op_sample_expert_actions = torch.zeros(num_samples, self.max_seq_len,self.actions_dim).type(torch.FloatTensor)

        op_sample_actions_shot_logit = []

        op_vae_lmbd_latent_mean = []
        op_vae_lmbd_latent_logvar = []

        
        """for decoded action saving"""
        
        context_actions = torch.zeros(num_samples, self.max_seq_len, self.actions_dim).type(torch.FloatTensor)
        op_context_actions = torch.zeros(num_samples, self.max_seq_len, self.actions_dim).type(torch.FloatTensor)
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
        bm = BrownianInterval(t0=0, t1=2*self.max_seq_len, size=(num_samples, self.state_embed_dim+1),\
        dtype=samples.dtype, device=samples.device, levy_area_approximation=LEVY_AREA_APPROXIMATIONS.none,\
        entropy= config['seed'], tol=1e-5, halfway_tree=True)
        step_size = self.step_size
        prev_t = curr_t = 0
        zs = []

        """loss for landing and moving position"""
        GMM_loss = 0
        regularization_loss = 0

        for i in range(self.max_seq_len):
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
                                player_samples[j, i, :] = inp_states.view(1,self.states_dim).data
                                if f_or_h ==  "h":
                                    if player_samples[j, i, 0] == 0 :
                                        stop_flag[j] = 1
                        else:
                            if f_or_h == "f":
                                stop_flag[j] = 1
                            else:
                                inp_states = get_next_player_state(op_sample_actions[j, i-1, :].data, opponent_samples[j, i-1, :].data, player_num)
                                inp_states = inp_states.cuda()
                                player_samples[j, i, :] = inp_states.view(1,self.states_dim).data
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
                            opponent_samples[j, i, :] = inp_states.view(1,self.states_dim).data
                            if f_or_h == "h":
                                if opponent_samples[j, i, 0] == 0 :
                                    op_stop_flag[j] = 1
                        else:
                            if f_or_h == "f":
                                op_stop_flag[j] = 1
                            else:
                                inp_states = get_next_player_state(sample_actions[j, i, :].data, player_samples[j, i, :].data, opponent_num)
                                inp_states = inp_states.cuda()
                                opponent_samples[j, i, :] = inp_states.view(1,self.states_dim).data
                                if opponent_samples[j, i, 0] == 0 :
                                    op_stop_flag[j] = 1
                
                """action projection layer"""

                if p == 0:
                    if f_or_h == "f":
                        z_f, f, z_h, h = self.forward(player_samples, expert_actions.copy(), player_samples[:, i, :], f_or_h)
                    else:
                        z_h, h = self.forward(player_samples, expert_actions.copy(), player_samples[:, i, :], f_or_h)

                    vae_lmbd_latent_mean.append(self.subspace_lmbd.latent_mean)
                    vae_lmbd_latent_logvar.append(self.subspace_lmbd.latent_logvar)
                    
                    # context decoder
                    context_land_logit = self.context_land_area(self._ctx)
                    context_shot_logit = self.context_shot_type(self._ctx)
                    context_move_logit = self.context_move_area(self._ctx)
                    context_actions_shot_logit.append(context_shot_logit)
                else:
                    if f_or_h == "f":
                        z_f, f, z_h, h = self.forward(opponent_samples, op_expert_actions.copy(), opponent_samples[:, i, :], f_or_h)
                    else:
                        z_h, h = self.forward(opponent_samples, op_expert_actions.copy(), opponent_samples[:, i, :], f_or_h)
                    op_vae_lmbd_latent_mean.append(self.subspace_lmbd.latent_mean)
                    op_vae_lmbd_latent_logvar.append(self.subspace_lmbd.latent_logvar)
                    
                    # context decoder                 
                    context_land_logit = self.context_land_area(self._ctx)
                    context_shot_logit = self.context_shot_type(self._ctx)
                    context_move_logit = self.context_move_area(self._ctx)
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
                    g = self.g(z_f)
                else:
                    g = self.g(z_h)
                
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
                land_logit = self.predict_land_area(project_embedding)
                shot_logit = self.predict_shot_type(project_embedding)
                move_logit = self.predict_move_area(project_embedding)
                
                land_logit[land_logit==0] = eps
                move_logit[move_logit==0] = eps

                land_gmm = D.MixtureSameFamily(D.Categorical(self.softmax(land_logit[:,:self.gmm_num])),\
                D.Independent(D.Normal(torch.reshape(land_logit[:,self.gmm_num:self.gmm_num*3],(-1,self.gmm_num,2)),\
                torch.reshape(torch.abs(torch.clamp(land_logit[:,self.gmm_num*3:],-self.gmm_thres,self.gmm_thres)),(-1,self.gmm_num,2))), 1))

                move_gmm = D.MixtureSameFamily(D.Categorical(self.softmax(move_logit[:,:self.gmm_num])),\
                D.Independent(D.Normal(torch.reshape(move_logit[:,self.gmm_num:self.gmm_num*3],(-1,self.gmm_num,2)),\
                torch.reshape(torch.abs(torch.clamp(move_logit[:,self.gmm_num*3:],-self.gmm_thres,self.gmm_thres)),(-1,self.gmm_num,2))), 1))

                agent_shot_type = torch.multinomial(torch.exp(self.log_softmax(shot_logit)), 1)
                while agent_shot_type[0, 0] == 0:
                    agent_shot_type = torch.multinomial(torch.exp(self.log_softmax(shot_logit)), 1)
                
                acts = torch.cat((land_gmm.sample(), agent_shot_type, move_gmm.sample()),1)

                """
                sample decoded action
                """
                context_land_logit[context_land_logit==0] = eps
                context_move_logit[context_move_logit==0] = eps

                context_land_gmm = D.MixtureSameFamily(D.Categorical(self.softmax(context_land_logit[:,:self.gmm_num])),\
                D.Independent(D.Normal(torch.reshape(context_land_logit[:,self.gmm_num:self.gmm_num*3],(-1,self.gmm_num,2)),\
                torch.reshape(torch.abs(torch.clamp(context_land_logit[:,self.gmm_num*3:],-self.gmm_thres,self.gmm_thres)),(-1,self.gmm_num,2))), 1))
                
                context_move_gmm = D.MixtureSameFamily(D.Categorical(self.softmax(context_move_logit[:,:self.gmm_num])),\
                D.Independent(D.Normal(torch.reshape(context_move_logit[:,self.gmm_num:self.gmm_num*3],(-1,self.gmm_num,2)),\
                torch.reshape(torch.abs(torch.clamp(context_move_logit[:,self.gmm_num*3:],-self.gmm_thres,self.gmm_thres)),(-1,self.gmm_num,2))), 1))

                context_shot_type = torch.multinomial(torch.exp(self.log_softmax(context_shot_logit)), 1)
                while context_shot_type[0, 0] == 0:
                    context_shot_type = torch.multinomial(torch.exp(self.log_softmax(context_shot_logit)), 1)
                
                context_acts = torch.cat((context_land_gmm.sample(), context_shot_type, context_move_gmm.sample()),1) 
                
                """
                1. save predicted action and decoded action for evaluation
                2. compute landing and moving loss (GMM_loss and regularization_loss)
                """
                if p == 0:
                    sample_actions[:, i, :] = acts.view(num_samples,self.actions_dim).data
                    context_actions[:, i, :] = context_acts.view(num_samples,self.actions_dim).data
                    sample_actions_shot_logit.append(shot_logit)

                    for j in range(eval_len):
                        if stop_flag[j] == 1:
                            sample_actions[j, i, :] = torch.zeros(self.actions_dim).type(torch.FloatTensor)
                        else:
                            if i < len(expert_actions[j]):
                                if expert_actions[j][i][:2].tolist() != [0,1]:
                                    land_gd = torch.reshape(land_logit[:,self.gmm_num:self.gmm_num*3],(-1,self.gmm_num,2))
                                    for r in range(self.gmm_num):
                                        regularization_loss += torch.cdist(land_gd[:,r,:], land_gd[:,r:,:], p=2).mean()
                                    move_gd = torch.reshape(move_logit[:,self.gmm_num:self.gmm_num*3],(-1,self.gmm_num,2))
                                    for r in range(self.gmm_num):
                                        regularization_loss += torch.cdist(move_gd[:,r,:], move_gd[:,r:,:], p=2).mean()
                                    GMM_loss += -land_gmm.log_prob(expert_actions[j][i][:2].cuda())[j]
                                    GMM_loss += -move_gmm.log_prob(expert_actions[j][i][3:].cuda())[j]

                        if check_tatic_next_status(context_actions[j, i, :].data):
                            ctx_stop_flag[j] = 1
                        
                        if ctx_stop_flag[j] == 1:
                            context_actions[j, i, :] = torch.zeros(self.actions_dim).type(torch.FloatTensor)
                        else:
                            if i < len(expert_actions[j]):
                                if expert_actions[j][i][:2].tolist() != [0,1]:
                                    context_land_gd = torch.reshape(context_land_logit[:,self.gmm_num:self.gmm_num*3],(-1,self.gmm_num,2))
                                    for r in range(self.gmm_num):
                                        regularization_loss += torch.cdist(context_land_gd[:,r,:], context_land_gd[:,r:,:], p=2).mean()
                                    context_move_gd = torch.reshape(context_move_logit[:,self.gmm_num:self.gmm_num*3],(-1,self.gmm_num,2))
                                    for r in range(self.gmm_num):
                                        regularization_loss += torch.cdist(context_move_gd[:,r,:], context_move_gd[:,r:,:], p=2).mean()
                                    GMM_loss += -context_land_gmm.log_prob(expert_actions[j][i][:2].cuda())[j]
                                    GMM_loss += -context_move_gmm.log_prob(expert_actions[j][i][3:].cuda())[j]
                    stop_flag = [0] * eval_len
                else:
                    op_sample_actions[:, i, :] = acts.view(num_samples,self.actions_dim).data
                    op_context_actions[:, i, :] = context_acts.view(num_samples,self.actions_dim).data
                    op_sample_actions_shot_logit.append(shot_logit)

                    for j in range(op_eval_len):
                        if op_stop_flag[j] == 1:
                            op_sample_actions[j, i, :] = torch.zeros(self.actions_dim).type(torch.FloatTensor)
                        else:
                            if i < len(op_expert_actions[j]):
                                if op_expert_actions[j][i][:2].tolist() != [0,1]:
                                    land_gd = torch.reshape(land_logit[:,self.gmm_num:self.gmm_num*3],(-1,self.gmm_num,2))
                                    for r in range(self.gmm_num):
                                        regularization_loss += torch.cdist(land_gd[:,r,:], land_gd[:,r:,:], p=2).mean()
                                    move_gd = torch.reshape(move_logit[:,self.gmm_num:self.gmm_num*3],(-1,self.gmm_num,2))
                                    for r in range(self.gmm_num):
                                        regularization_loss += torch.cdist(move_gd[:,r,:], move_gd[:,r:,:], p=2).mean()
                                    GMM_loss += -land_gmm.log_prob(op_expert_actions[j][i][:2].cuda())[j]
                                    GMM_loss += -move_gmm.log_prob(op_expert_actions[j][i][3:].cuda())[j]
                        
                        if check_tatic_next_status(op_context_actions[j, i, :].data):
                            op_ctx_stop_flag[j] = 1
                        
                        if op_ctx_stop_flag[j] == 1:
                            op_context_actions[j, i, :] = torch.zeros(self.actions_dim).type(torch.FloatTensor)
                        else:
                            if i < len(op_expert_actions[j]):
                                if op_expert_actions[j][i][:2].tolist() != [0,1]:
                                    context_land_gd = torch.reshape(context_land_logit[:,self.gmm_num:self.gmm_num*3],(-1,self.gmm_num,2))
                                    for r in range(self.gmm_num):
                                        regularization_loss += torch.cdist(context_land_gd[:,r,:], context_land_gd[:,r:,:], p=2).mean()
                                    context_move_gd = torch.reshape(context_move_logit[:,self.gmm_num:self.gmm_num*3],(-1,self.gmm_num,2))
                                    for r in range(self.gmm_num):
                                        regularization_loss += torch.cdist(context_move_gd[:,r,:], context_move_gd[:,r:,:], p=2).mean()
                                    GMM_loss += -context_land_gmm.log_prob(op_expert_actions[j][i][:2].cuda())[j]
                                    GMM_loss += -context_move_gmm.log_prob(op_expert_actions[j][i][3:].cuda())[j]
                    op_stop_flag = [0] * op_eval_len

        """re-choose context"""
        self._ctx = None
        
        """compute SDEs_KLD"""
        zs = torch.stack(zs, dim=0)
        zs, log_ratio = zs.split(split_size=(z0.size(1) - 1, 1), dim=2)
        log_ratio_increments = torch.stack([log_ratio_t_plus_1 - log_ratio_t for log_ratio_t_plus_1, log_ratio_t in zip(log_ratio[1:], log_ratio[:-1])], dim=0).squeeze(dim=2)
        SDEs_KLD = log_ratio_increments.sum(dim=0).mean(dim=0)
        SDEs_KLD += (GMM_loss/self.gmm_num)*(1-self.alpha)*0.01
        SDEs_KLD -= (regularization_loss/self.gmm_num)*self.alpha*0.01
        
        return sample_actions, sample_expert_actions,\
        sample_actions_shot_logit,\
        op_sample_actions, op_sample_expert_actions,\
        op_sample_actions_shot_logit,\
        record, vae_lmbd_latent_mean, vae_lmbd_latent_logvar, op_vae_lmbd_latent_mean, op_vae_lmbd_latent_logvar, SDEs_KLD,\
        context_actions, context_actions_shot_logit,\
        op_context_actions, op_context_actions_shot_logit
        
    def predict_batchNLLLoss(self, inp, target, shot_d, mean_d, logvar_d):
        batch_size, seq_len = inp.size()[:2]
        loss = 0

        ce_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

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
                
                if stp_list[j] == 0:
                    loss += ce_fn(torch.unsqueeze(shot_d[i][j],dim=0), torch.unsqueeze(target.data[j, i, 2].type(torch.LongTensor).cuda(),dim=0))
                    loss += (-0.5 * torch.mean(1 + torch.unsqueeze(logvar_d[i],dim=0) - torch.unsqueeze(mean_d[i],dim=0).pow(2) - torch.unsqueeze(logvar_d[i],dim=0).exp()))/batch_size
                    
        GLD = [0]*20
        ALD = [0]*20
        for length in gen_list:
            GLD[length] += 1
        for length in ans_list:
            ALD[length] += 1
        
        KLD = kl_divergence(GLD, ALD)

        length_error = np.absolute((np.asarray(gen_list) - np.asarray(ans_list)))
        MLD = np.sum(length_error) / len(ans_list)
        loss += KLD
        loss += MLD*len(ans_list)

        return loss, len_list

    def decode_batchNLLLoss(self, inp, target, shot_d):
        batch_size, seq_len = inp.size()[:2]
        loss = 0

        ce_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        
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
                
                if stp_list[j] == 0:
                    loss += ce_fn(torch.unsqueeze(shot_d[i][j],dim=0), torch.unsqueeze(target.data[j, i, 2].type(torch.LongTensor).cuda(),dim=0))
                    
        GLD = [0]*20
        ALD = [0]*20
        for length in gen_list:
            GLD[length] += 1
        for length in ans_list:
            ALD[length] += 1
        KLD = kl_divergence(GLD, ALD)

        length_error = np.absolute((np.asarray(gen_list) - np.asarray(ans_list)))
        MLD = np.sum(length_error) / len(ans_list)
        loss += KLD
        loss += MLD*len(ans_list)

        return loss

    def eval_batchNLLLoss(self, inp, target, shot_d):
        batch_size, seq_len = inp.size()[:2]

        shot_ce = 0
        land_mae = 0
        move_mae = 0
        land_dtw = 0
        move_dtw = 0

        ce_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        mae_fn = nn.L1Loss(reduction='sum')
        
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
            
                if stp_list[j] == 0:
                    land_mae += mae_fn(inp.data[j, i, 0:2], target.data[j, i, 0:2])
                    move_mae += mae_fn(inp.data[j, i, 3:], target.data[j, i, 3:])
                    shot_ce += ce_fn(shot_d[i], torch.unsqueeze(target.data[j, i, 2].type(torch.LongTensor).cuda(),dim=0))
                    
            land_dtw += dtw_ndim.distance(inp.data[j, :, :2].detach().cpu().numpy(), target.data[j, :, :2].cpu().numpy())
            move_dtw += dtw_ndim.distance(inp.data[j, :, 3:].detach().cpu().numpy(), target.data[j, :, 3:].cpu().numpy())
                
        return shot_ce, land_mae, land_dtw, move_mae, move_dtw, len_list, gen_list, ans_list