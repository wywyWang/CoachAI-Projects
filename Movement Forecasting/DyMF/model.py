import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch import Tensor

from typing import Dict, Optional
PAD = 0
# make adjacency matrix according to encode length which number of row is encode length times 2
def initialize_adjacency_matrix(batch_size, encode_length, shot_type):
    adjacency_matrix = torch.zeros((13, encode_length * 2, encode_length * 2), dtype=int).to(shot_type.device)

    for row in range(encode_length * 2):
        node_index = int(row / 2)
        if row % 2 == 0: # black node
            if node_index % 2 == 0: # even black node
                if (node_index + 1) * 2 <= encode_length * 2 - 1:
                    adjacency_matrix[11][row][(node_index + 1) * 2] = 1
                    adjacency_matrix[11][(node_index + 1) * 2][row] = 1
            if node_index % 2 == 1: # odd black node
                if (node_index + 1) * 2 <= encode_length * 2 - 1:
                    adjacency_matrix[12][row][(node_index + 1) * 2] = 1
                    adjacency_matrix[12][(node_index + 1) * 2][row] = 1
        if row % 2 == 1: # white node
            if node_index % 2 == 0: # even white node
                if (node_index + 1) * 2 + 1 <= encode_length * 2 - 1:
                    adjacency_matrix[12][row][(node_index + 1) * 2 + 1] = 1
                    adjacency_matrix[12][(node_index + 1) * 2 + 1][row] = 1
            if node_index % 2 == 1: # odd white node                
                if (node_index + 1) * 2 + 1 <= encode_length * 2 - 1:
                    adjacency_matrix[11][row][(node_index + 1) * 2 + 1] = 1
                    adjacency_matrix[11][(node_index + 1) * 2 + 1][row] = 1
    
    # for row in range(2, encode_length * 2, 2):
    #     adjacency_matrix[11][row-2][row-1] = 1
    #     adjacency_matrix[11][row-1][row-2] = 1
    
    adjacency_matrix = torch.tile(adjacency_matrix, (batch_size, 1, 1, 1))
    for batch in range(batch_size):
        for step in range(len(shot_type[batch])):
            if step % 2 == 0:
                adjacency_matrix[batch][shot_type[batch][step]][step * 2][(step + 1) * 2 + 1] = 1
                adjacency_matrix[batch][shot_type[batch][step]][(step + 1) * 2 + 1][step * 2] = 1
            if step % 2 == 1:
                adjacency_matrix[batch][shot_type[batch][step]][(step * 2) + 1][(step + 1) * 2] = 1
                adjacency_matrix[batch][shot_type[batch][step]][(step + 1) * 2][(step * 2) + 1] = 1

    return adjacency_matrix

def update_adjacency_matrix(batch_size, step, adjacency_matrix, shot_type_predict=False):
    new_adjacency_matrix = torch.zeros((batch_size, 13, step * 2, step * 2), dtype=int).to(adjacency_matrix.device)
    new_adjacency_matrix[:, :, :-2, :-2] = adjacency_matrix
    adjacency_matrix = new_adjacency_matrix.clone()
        
    for row in range(step * 2):
        node_index = int(row / 2)
        if row % 2 == 0: # black node
            if node_index % 2 == 0: # even black node
                if (node_index + 1) * 2 <= step * 2 - 1:
                    adjacency_matrix[:, 11, row, (node_index + 1) * 2] = 1
                    adjacency_matrix[:, 11, (node_index + 1) * 2, row] = 1
            if node_index % 2 == 1: # odd black node
                if (node_index + 1) * 2 <= step * 2 - 1:
                    adjacency_matrix[:, 12, row, (node_index + 1) * 2] = 1
                    adjacency_matrix[:, 12, (node_index + 1) * 2, row] = 1
        if row % 2 == 1: # white node
            if node_index % 2 == 0: # even white node
                if (node_index + 1) * 2 + 1 <= step * 2 - 1:
                    adjacency_matrix[:, 12, row, (node_index + 1) * 2 + 1] = 1
                    adjacency_matrix[:, 12, (node_index + 1) * 2 + 1, row] = 1
            if node_index % 2 == 1: # odd white node                
                if (node_index + 1) * 2 + 1 <= step * 2 - 1:
                    adjacency_matrix[:, 11, row, (node_index + 1) * 2 + 1] = 1
                    adjacency_matrix[:, 11, (node_index + 1) * 2 + 1, row] = 1

    # for row in range(2, step * 2, 2):
    #     adjacency_matrix[:, 11, row-2, row-1] = 1
    #     adjacency_matrix[:, 11, row-1, row-2] = 1

    # if shot_type_predict:
    #     if step % 2 == 0:
    #         adjacency_matrix[:, 11, (step - 2) * 2, (step - 1) * 2 + 1] = 1  
    #         adjacency_matrix[:, 11, (step - 1) * 2 + 1, (step - 2) * 2] = 1 
    #     if step % 2 == 1:
    #         adjacency_matrix[:, 11, (step - 2) * 2 + 1, (step - 1) * 2] = 1  
    #         adjacency_matrix[:, 11, (step - 1) * 2, (step - 2) * 2 + 1] = 1

    if shot_type_predict:        
        if step % 2 == 0:
            adjacency_matrix[:, :, (step - 1) * 2, :] = 0
            adjacency_matrix[:, :, :, (step - 1) * 2] = 0
        if step % 2 == 1: 
            adjacency_matrix[:, :, (step - 1) * 2 + 1, :] = 0
            adjacency_matrix[:, :, :, (step - 1) * 2 + 1] = 0
   
    return adjacency_matrix

def preprocess_adj(A):
    I = torch.eye(A.size(1)).unsqueeze(0).to(A.device)
    A_hat = A + I 
    D_hat_diag = torch.sum(A_hat, dim=2)
    D_hat_diag_inv_sqrt = torch.pow(D_hat_diag, -0.5)

    D_hat_diag_inv_sqrt[torch.isinf(D_hat_diag_inv_sqrt)] = 0.
   
    b = torch.eye(D_hat_diag_inv_sqrt.size(1)).to(A.device)
    c = D_hat_diag_inv_sqrt.unsqueeze(2).expand(*D_hat_diag_inv_sqrt.size(), D_hat_diag_inv_sqrt.size(1))
    D_hat_inv_sqrt = c * b
    
    preprocess_A = torch.matmul(torch.matmul(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)

    return preprocess_A

def create_src_lengths_mask(
        batch_size: int, src_lengths: Tensor, max_src_len: Optional[int] = None
):
    """
    Generate boolean mask to prevent attention beyond the end of source
    Inputs:
      batch_size : int
      src_lengths : [batch_size] of sentence lengths
      max_src_len: Optionally override max_src_len for the mask
    Outputs:
      [batch_size, max_src_len]
    """
    if max_src_len is None:
        max_src_len = int(src_lengths.max())
    src_indices = torch.arange(0, max_src_len).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_src_len)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_src_len)
 
    # returns [batch_size, max_seq_len]
    return (src_indices < src_lengths).int().detach()
 
 
def masked_softmax(scores, src_lengths, src_length_masking=True):
    """Apply source length masking then softmax.
    Input and output have shape bsz x src_len"""
    if src_length_masking:
        bsz, max_src_len = scores.size()
        # print('bsz:', bsz)
        # compute masks
        src_mask = create_src_lengths_mask(bsz, src_lengths)
        # Fill pad positions with -inf
        scores = scores.masked_fill(src_mask == 0, -np.inf)
 
    # Cast to float and then back again to prevent loss explosion under fp16.
    return F.softmax(scores.float(), dim=-1).type_as(scores)
 
 
class ParallelCoAttentionNetwork(nn.Module):
 
    def __init__(self, hidden_dim, co_attention_dim, src_length_masking=True):
        super(ParallelCoAttentionNetwork, self).__init__()
 
        self.hidden_dim = hidden_dim
        self.co_attention_dim = co_attention_dim
        self.src_length_masking = src_length_masking

        self.W_b = torch.nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        nn.init.xavier_uniform_(self.W_b, gain=nn.init.calculate_gain('relu'))
        self.W_v = torch.nn.Parameter(torch.Tensor(self.co_attention_dim, self.hidden_dim))
        nn.init.xavier_uniform_(self.W_v, gain=nn.init.calculate_gain('relu'))
        self.W_q = torch.nn.Parameter(torch.Tensor(self.co_attention_dim, self.hidden_dim))
        nn.init.xavier_uniform_(self.W_q, gain=nn.init.calculate_gain('relu'))
        self.w_hv = torch.nn.Parameter(torch.Tensor(self.co_attention_dim, 1))
        nn.init.xavier_uniform_(self.w_hv, gain=nn.init.calculate_gain('relu'))
        self.w_hq = torch.nn.Parameter(torch.Tensor(self.co_attention_dim, 1))
        nn.init.xavier_uniform_(self.w_hq, gain=nn.init.calculate_gain('relu'))

        # self.dropout = nn.Dropout(0.1)
        # self.W_b = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        # self.W_v = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        # self.W_q = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        # self.w_hv = nn.Parameter(torch.randn(self.co_attention_dim, 1))
        # self.w_hq = nn.Parameter(torch.randn(self.co_attention_dim, 1))
 
    def forward(self, V, Q, Q_lengths):
        """
        :param V: batch_size * hidden_dim * region_num, eg B x 512 x 196
        :param Q: batch_size * seq_len * hidden_dim, eg B x L x 512
        :param Q_lengths: batch_size
        :return:batch_size * 1 * region_num, batch_size * 1 * seq_len,
        batch_size * hidden_dim, batch_size * hidden_dim
        """
        # (batch_size, seq_len, region_num)
        C = torch.matmul(Q, torch.matmul(self.W_b, V))
        # (batch_size, co_attention_dim, region_num)
        H_v = nn.Tanh()(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))
        # (batch_size, co_attention_dim, seq_len)
        H_q = nn.Tanh()(
            torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))

        # (batch_size, 1, region_num)
        a_v = F.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)
        # (batch_size, 1, seq_len)
        a_q = F.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)
        # # (batch_size, 1, seq_len)

        # a_v = self.dropout(a_v)
        # a_q = self.dropout(a_q)
        
        masked_a_q = masked_softmax(
            a_q.squeeze(1), Q_lengths, self.src_length_masking
        ).unsqueeze(1)
 
        # (batch_size, hidden_dim)
        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1)))
        # (batch_size, hidden_dim)
        q = torch.squeeze(torch.matmul(masked_a_q, Q))
 
        return a_v, masked_a_q, v, q

class GCNDynamicLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, args, device):
        super(GCNDynamicLayer, self).__init__()

        self.device = device
        self.hidden_size = args['hidden_size']
        # self.weight_generate_weights = torch.nn.Parameter(torch.Tensor((args['max_length']+1)*args['hidden_size'], args['hidden_size'] * args['hidden_size']))
        # nn.init.xavier_uniform_(self.weight_generate_weights, gain=nn.init.calculate_gain('relu'))
        self.dropout = nn.Dropout(args['dropout'])
        self.lstm = nn.LSTM(args['hidden_size'], args['hidden_size'], num_layers=1, batch_first=True)
        self.conv1d = nn.Conv1d(args['hidden_size'], args['hidden_size'], 3, padding=1)

    def forward(self, node_embedding, adjacency_matrix, activation_function=None):
        if node_embedding.size(1) < self.hidden_size:
            padding = torch.zeros((node_embedding.size(0), self.hidden_size - node_embedding.size(1), node_embedding.size(2))).to(node_embedding.device)
            # padding = node_embedding[:, -1:, :].repeat([1, self.hidden_size - node_embedding.size(1), 1])
            dynamic_filter_input = torch.cat((node_embedding, padding), dim=1)
        else:
            dynamic_filter_input = node_embedding[:, node_embedding.size(1)-self.hidden_size:, :]

        #dynamic_filter_input = node_embedding.view(node_embedding.size(0), -1)
    

        #weight_generate_weights = self.weight_generate_weights[:dynamic_filter_input.size(1), :]

        #dynamic_filter = torch.matmul(dynamic_filter_input, weight_generate_weights).view(node_embedding.size(0), -1, node_embedding.size(2))
        dynamic_filter = self.conv1d(dynamic_filter_input)

        hidden = torch.zeros((1, node_embedding.size(0), self.hidden_size))
        # hidden = torch.empty(1, node_embedding.size(0), self.hidden_size)
        # hidden = nn.init.xavier_uniform_(hidden, gain=nn.init.calculate_gain('relu'))
        hidden = hidden.to(self.device)
        cell = torch.zeros((1, node_embedding.size(0), self.hidden_size))
        # cell = torch.empty(1, node_embedding.size(0), self.hidden_size)
        # cell = nn.init.xavier_uniform_(cell, gain=nn.init.calculate_gain('relu'))
        cell = cell.to(self.device)

        linear_weights, (hidden, cell) = self.lstm(dynamic_filter, (hidden, cell))        
        node_embedding = torch.matmul(adjacency_matrix, node_embedding)

        output = torch.matmul(node_embedding, linear_weights)
        output = activation_function(output)
        output = self.dropout(output)

        return output

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, num_layer, args, device):
        super(GCN, self).__init__()
        self.gcn_dynamic_layer_list = nn.ModuleList([GCNDynamicLayer(input_dim, hidden_dim, dropout, args, device) for _ in range(1)])
        
        self.hidden_activation_function = nn.ReLU()
        self.output_activation_function = nn.Sigmoid()

    def forward(self, node_embedding, adjacency_matrix):

        adjacency_matrix = preprocess_adj(adjacency_matrix).float().to(node_embedding.device)
        for i, gcn_dynamic_layer in enumerate(self.gcn_dynamic_layer_list):
            if i == len(self.gcn_dynamic_layer_list) - 1:
                node_embedding = gcn_dynamic_layer(node_embedding, adjacency_matrix, self.output_activation_function)
            else:
                node_embedding = gcn_dynamic_layer(node_embedding, adjacency_matrix, self.hidden_activation_function)
        return node_embedding

class relational_GCN_layer(nn.Module):
    def __init__(self, hidden_size, type_num, num_basis, device):
        super(relational_GCN_layer, self).__init__()
        # relation index should minus 1, because padding 0
        self.num_basis = num_basis
        self.hidden_size = hidden_size
        self.device = device
        self.type_num = type_num
        self.self_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.normalization_constant = torch.nn.Parameter(torch.Tensor(1, type_num - 1 + 3))

        self.basis_matrix = torch.nn.Parameter(torch.Tensor(num_basis, hidden_size, hidden_size))
        # self.bias = torch.nn.Parameter(torch.Tensor(type_num + 3, hidden_size)).to(device)
        self.linear_combination = torch.nn.Parameter(torch.Tensor(type_num - 1 + 2, num_basis))
        self.dropout = nn.Dropout(0.1)
        nn.init.xavier_uniform_(self.basis_matrix, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.linear_combination, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.normalization_constant, gain=nn.init.calculate_gain('relu'))
        
    def forward(self, node_embedding, adjacency_matrix, activation_function):
        mutil_relational_weight = torch.matmul(self.linear_combination, self.basis_matrix.view(self.num_basis, -1)).view(self.type_num - 1 + 2, self.hidden_size, self.hidden_size)

        adjacency_matrix = adjacency_matrix[:, 1:, :, :]
        connected_node_embedding = torch.matmul(adjacency_matrix.float(), node_embedding.unsqueeze(1))

        # connected_node_embedding = torch.sum(torch.matmul(connected_node_embedding, mutil_relational_weight.unsqueeze(0)) * self.normalization_constant.view(1, self.type_num - 1 + 3, 1, 1), dim=1)
        connected_node_embedding = torch.sum(torch.matmul(connected_node_embedding, mutil_relational_weight.unsqueeze(0)), dim=1)
        connected_node_embedding = connected_node_embedding + self.self_linear(node_embedding)

        output = activation_function(connected_node_embedding)
        output = self.dropout(output)
        return output

class relational_GCN(nn.Module):
    def __init__(self, hidden_size, type_num, num_basis, num_layer, device):
        super(relational_GCN, self).__init__()
        # relation index should minus 1, because padding 0

        self.num_basis = num_basis
        self.hidden_size = hidden_size
        self.device = device
        self.type_num = type_num

        self.hidden_activation_function = nn.ReLU()
        self.output_activation_function = nn.Sigmoid()

        self.rgcn_layer_list = nn.ModuleList([relational_GCN_layer(hidden_size, type_num, num_basis, device) for _ in range(num_layer)])

    def forward(self, node_embedding, adjacency_matrix):
        for i, rgcn_layer in enumerate(self.rgcn_layer_list):
            if i == len(self.rgcn_layer_list) - 1:
                node_embedding = rgcn_layer(node_embedding, adjacency_matrix, self.output_activation_function)
            else:
                node_embedding = rgcn_layer(node_embedding, adjacency_matrix, self.hidden_activation_function)
        return node_embedding     

class Decoder(nn.Module):
    def __init__(self, args, device):
        super(Decoder, self).__init__()
        player_num = args['player_num']
        player_dim = args['player_dim']

        type_num = args['type_num']

        location_dim = args['location_dim']

        hidden_size = args['hidden_size']

        num_layer = args['num_layer']

        self.player_num = args['player_num']
        self.type_num = args['type_num']

        self.player_embedding = nn.Embedding(player_num, player_dim)
        self.coordination_transform = nn.Linear(2, location_dim)

        self.model_input_linear = nn.Linear(player_dim + location_dim, hidden_size)

        self.rGCN = relational_GCN(hidden_size, type_num, args['num_basis'], num_layer, device)
        self.gcn = GCN(args['hidden_size'], args['hidden_size'], 0.1, num_layer, args, device)

        self.predict_shot_type = nn.Linear(hidden_size * 2, type_num)
        self.predict_xy = nn.Linear(hidden_size*2, 10)

        self.rgcn_weight = nn.Linear(args['hidden_size'], 1)
        self.gcn_weight = nn.Linear(args['hidden_size'], 1)

        self.partial_adjacency_matrix = torch.ones((args['max_length']+1, args['max_length']+1), dtype=int) - torch.eye(args['max_length']+1, dtype=int)

        self.sigmoid = nn.Sigmoid()
        
        self.co_attention = ParallelCoAttentionNetwork(args['hidden_size'], args['hidden_size'], src_length_masking=False)
        self.co_attention_linear_A = nn.Linear(args['hidden_size'], 1)
        self.co_attention_linear_B = nn.Linear(args['hidden_size'], 1)

        self.original_weight = nn.Linear(args['hidden_size']*2, 1)
        self.passed_weight = nn.Linear(args['hidden_size']*2, 1)

        self.linear_for_dynmaic_gcn = nn.Linear(args['hidden_size'] + args['player_dim'], args['hidden_size'])

    def forward(self, player, step, encode_node_embedding, original_embedding, adjacency_matrix, 
                player_A_x, player_A_y, player_B_x, player_B_y, 
                shot_type=None, train=False, first=False):
        batch_size = player.size(0)
        player_embedding = self.player_embedding(player)

        prev_player_A_coordination = torch.cat((player_A_x.unsqueeze(2), player_A_y.unsqueeze(2)), dim=2).float()
        prev_player_B_coordination = torch.cat((player_B_x.unsqueeze(2), player_B_y.unsqueeze(2)), dim=2).float()

        prev_coordination_sequence = torch.stack((prev_player_A_coordination, prev_player_B_coordination), dim=2).view(player.size(0), -1, 2)
        prev_coordination_transform = self.coordination_transform(prev_coordination_sequence)
        prev_coordination_transform = F.relu(prev_coordination_transform)
        rally_information = torch.cat((prev_coordination_transform, player_embedding), dim=-1)
        initial_embedding = self.model_input_linear(rally_information)

        if not first:            
            player_A_coordination = torch.cat((player_A_x.unsqueeze(2), player_A_y.unsqueeze(2)), dim=2).float()
            player_B_coordination = torch.cat((player_B_x.unsqueeze(2), player_B_y.unsqueeze(2)), dim=2).float()
            coordination_sequence = torch.stack((player_A_coordination, player_B_coordination), dim=2).view(player.size(0), -1, 2)
            coordination_transform = self.coordination_transform(coordination_sequence)
            coordination_transform = F.relu(coordination_transform)
            model_input = torch.cat((coordination_transform, player_embedding), dim=-1)
            model_input = self.model_input_linear(model_input)

            # original_weight = self.sigmoid(self.original_weight(original_embedding[:, -2:, :].view(batch_size, 1, -1)))
            # passed_weight = self.sigmoid(self.passed_weight(encode_node_embedding[:, -2:, :].view(batch_size, 1, -1)))
            # tmp = original_embedding * original_weight + passed_weight * encode_node_embedding
            
            # original_embedding = torch.cat((original_embedding, model_input), dim=1)

            model_input = torch.cat((encode_node_embedding, model_input), dim=1)

            player_A_embedding = model_input[:, 0::2, :].clone()
            player_B_embedding = model_input[:, 1::2, :].clone()
            partial_adjacency_matrix = self.partial_adjacency_matrix[:player_A_embedding.size(1), :player_A_embedding.size(1)].to(player.device)
            partial_adjacency_matrix = torch.tile(partial_adjacency_matrix, (batch_size, 1, 1))

            dynamic_gcn_input_A = torch.cat((player_A_embedding, player_embedding[:, 0:1, :].repeat(1, player_A_embedding.size(1), 1)), dim=-1)
            dynamic_gcn_input_B = torch.cat((player_B_embedding, player_embedding[:, 1:2, :].repeat(1, player_B_embedding.size(1), 1)), dim=-1)

            dynamic_gcn_input_A = self.linear_for_dynmaic_gcn(dynamic_gcn_input_A)
            dynamic_gcn_input_B = self.linear_for_dynmaic_gcn(dynamic_gcn_input_B)

            player_A_node_embedding = self.gcn(dynamic_gcn_input_A, partial_adjacency_matrix)
            player_B_node_embedding = self.gcn(dynamic_gcn_input_B, partial_adjacency_matrix)
            
            full_graph_node_embedding = self.rGCN(model_input, adjacency_matrix)
            tmp_embedding = torch.zeros((full_graph_node_embedding.size(0), full_graph_node_embedding.size(1), full_graph_node_embedding.size(2))).to(player.device)
            
            _, _, A_weight, B_weight =  self.co_attention(player_A_node_embedding.permute(0, 2, 1), player_B_node_embedding, batch_size)
            A_weight = self.sigmoid(self.co_attention_linear_A(A_weight))
            B_weight = self.sigmoid(self.co_attention_linear_B(B_weight))

            player_A_node_embedding = player_A_node_embedding + B_weight.unsqueeze(1) * player_B_node_embedding
            player_B_node_embedding = player_B_node_embedding + A_weight.unsqueeze(1) * player_A_node_embedding

            rgcn_embedding_A = full_graph_node_embedding[:, 0::2, :].clone()[:, -1:, :].view(batch_size, -1)
            rgcn_embedding_B = full_graph_node_embedding[:, 1::2, :].clone()[:, -1:, :].view(batch_size, -1)
            gcn_embedding_A = player_A_node_embedding[:, -1:, :].clone().view(batch_size, -1)
            gcn_embedding_B = player_B_node_embedding[:, -1:, :].clone().view(batch_size, -1)

            rgcn_weight_A = self.rgcn_weight(rgcn_embedding_A)
            rgcn_weight_B = self.rgcn_weight(rgcn_embedding_B)
            gcn_weight_A = self.gcn_weight(gcn_embedding_A)
            gcn_weight_B = self.gcn_weight(gcn_embedding_B)
            
            w_rgcn_A = self.sigmoid(rgcn_weight_A)
            w_gcn_A = self.sigmoid(gcn_weight_A)
            w_rgcn_B = self.sigmoid(rgcn_weight_B)
            w_gcn_B = self.sigmoid(gcn_weight_B)

            tmp_embedding[:, 0::2, :] = full_graph_node_embedding[:, 0::2, :] * w_rgcn_A.unsqueeze(1) + player_A_node_embedding * w_gcn_A.unsqueeze(1)
            tmp_embedding[:, 1::2, :] = full_graph_node_embedding[:, 1::2, :] * w_rgcn_B.unsqueeze(1) + player_B_node_embedding * w_gcn_B.unsqueeze(1)

            passed_node_embedding = torch.cat((encode_node_embedding[:, :-2, :], tmp_embedding[:, -4:, :]), dim=1)
        else:
            passed_node_embedding = encode_node_embedding.clone()

        tmp_adjacency_matrix = update_adjacency_matrix(batch_size, step, adjacency_matrix, shot_type_predict=True)

        # original_weight = self.sigmoid(self.original_weight(original_embedding[:, -2:, :].view(batch_size, 1, -1)))
        # passed_weight = self.sigmoid(self.passed_weight(passed_node_embedding[:, -2:, :].view(batch_size, 1, -1)))
        # tmp = original_weight * original_embedding + passed_weight * passed_node_embedding
        model_input = torch.cat((passed_node_embedding, initial_embedding), dim=1)

        # =============================================================================================================================================
        tmp_model_input = model_input[:, :-2, :]
        player_A_embedding = tmp_model_input[:, 0::2, :].clone()
        player_B_embedding = tmp_model_input[:, 1::2, :].clone()

        if step % 2 == 0:            
            player_B_embedding = torch.cat((player_B_embedding, model_input[:, (step-1)*2+1:(step-1)*2+1+1, :]), dim=1)
        if step % 2 == 1:
            player_A_embedding = torch.cat((player_A_embedding, model_input[:, (step-1)*2:(step-1)*2+1, :]), dim=1)
        
        partial_adjacency_matrix_A = self.partial_adjacency_matrix[:player_A_embedding.size(1), :player_A_embedding.size(1)].to(player.device)
        partial_adjacency_matrix_B = self.partial_adjacency_matrix[:player_B_embedding.size(1), :player_B_embedding.size(1)].to(player.device)
        
        partial_adjacency_matrix_A = torch.tile(partial_adjacency_matrix_A, (batch_size, 1, 1))
        partial_adjacency_matrix_B = torch.tile(partial_adjacency_matrix_B, (batch_size, 1, 1))

        dynamic_gcn_input_A = torch.cat((player_A_embedding, player_embedding[:, 0:1, :].repeat(1, player_A_embedding.size(1), 1)), dim=-1)
        dynamic_gcn_input_B = torch.cat((player_B_embedding, player_embedding[:, 1:2, :].repeat(1, player_B_embedding.size(1), 1)), dim=-1)

        dynamic_gcn_input_A = self.linear_for_dynmaic_gcn(dynamic_gcn_input_A)
        dynamic_gcn_input_B = self.linear_for_dynmaic_gcn(dynamic_gcn_input_B)

        player_A_node_embedding = self.gcn(dynamic_gcn_input_A, partial_adjacency_matrix_A)
        player_B_node_embedding = self.gcn(dynamic_gcn_input_B, partial_adjacency_matrix_B)

        if step % 2 == 0:            
            tmp_model_input = model_input[:, torch.arange(model_input.size(1))!=(step-1)*2]
            tmp_adjacency_matrix = tmp_adjacency_matrix[:, :, torch.arange(tmp_adjacency_matrix.size(2))!=(step-1)*2]
            tmp_adjacency_matrix = tmp_adjacency_matrix[:, :, :, torch.arange(tmp_adjacency_matrix.size(3))!=(step-1)*2]
        if step % 2 == 1:
            tmp_model_input = model_input[:, torch.arange(model_input.size(1))!=(step-1)*2+1]
            tmp_adjacency_matrix = tmp_adjacency_matrix[:, :, torch.arange(tmp_adjacency_matrix.size(2))!=(step-1)*2+1]
            tmp_adjacency_matrix = tmp_adjacency_matrix[:, :, :, torch.arange(tmp_adjacency_matrix.size(3))!=(step-1)*2+1]

        full_graph_node_embedding = self.rGCN(tmp_model_input, tmp_adjacency_matrix)
        padding_full_graph_node_embedding = torch.zeros((full_graph_node_embedding .size(0), full_graph_node_embedding.size(1)+1, full_graph_node_embedding .size(2))).to(player.device)

        if step % 2 == 0:
            padding_full_graph_node_embedding[:, :-2, :] = full_graph_node_embedding[:, :-1, :]
            padding_full_graph_node_embedding[:, -1, :] = full_graph_node_embedding[:, -1, :]  
        if step % 2 == 1: 
            padding_full_graph_node_embedding[:, :-1, :] = full_graph_node_embedding

        full_graph_node_embedding = padding_full_graph_node_embedding.clone()
        tmp_embedding = torch.zeros((full_graph_node_embedding.size(0), full_graph_node_embedding.size(1), full_graph_node_embedding.size(2))).to(player.device)

        _, _, A_weight, B_weight =  self.co_attention(player_A_node_embedding.permute(0, 2, 1), player_B_node_embedding, batch_size)
        A_weight = self.sigmoid(self.co_attention_linear_A(A_weight))
        B_weight = self.sigmoid(self.co_attention_linear_B(B_weight))

        if step % 2 == 0:            
            player_A_node_embedding = torch.cat((player_A_node_embedding, torch.zeros((player_A_node_embedding.size(0), 1, player_A_node_embedding.size(2))).to(player.device)), dim=1)
        if step % 2 == 1:
            player_B_node_embedding = torch.cat((player_B_node_embedding, torch.zeros((player_B_node_embedding.size(0), 1, player_B_node_embedding.size(2))).to(player.device)), dim=1)

        player_A_node_embedding = player_A_node_embedding + B_weight.unsqueeze(1) * player_B_node_embedding
        player_B_node_embedding = player_B_node_embedding + A_weight.unsqueeze(1) * player_A_node_embedding

        if step % 2 == 0:
            rgcn_embedding_A = full_graph_node_embedding[:, 0::2, :].clone()[:, -2:-1, :].view(batch_size, -1)
            rgcn_embedding_B = full_graph_node_embedding[:, 1::2, :].clone()[:, -1:, :].view(batch_size, -1)
            gcn_embedding_A = player_A_node_embedding[:, -2:-1, :].clone().view(batch_size, -1)
            gcn_embedding_B = player_B_node_embedding[:, -1:, :].clone().view(batch_size, -1)
        if step % 2 == 1: 
            rgcn_embedding_A = full_graph_node_embedding[:, 0::2, :].clone()[:, -1:, :].view(batch_size, -1)
            rgcn_embedding_B = full_graph_node_embedding[:, 1::2, :].clone()[:, -2:-1, :].view(batch_size, -1)
            gcn_embedding_A = player_A_node_embedding[:, -1:, :].clone().view(batch_size, -1)
            gcn_embedding_B = player_B_node_embedding[:, -2:-1, :].clone().view(batch_size, -1)
        
        rgcn_weight_A = self.rgcn_weight(rgcn_embedding_A)
        rgcn_weight_B = self.rgcn_weight(rgcn_embedding_B)
        gcn_weight_A = self.gcn_weight(gcn_embedding_A)
        gcn_weight_B = self.gcn_weight(gcn_embedding_B)
        
        w_rgcn_A = self.sigmoid(rgcn_weight_A)
        w_gcn_A = self.sigmoid(gcn_weight_A)
        w_rgcn_B = self.sigmoid(rgcn_weight_B)
        w_gcn_B = self.sigmoid(gcn_weight_B)

        tmp_embedding[:, 0::2, :] = full_graph_node_embedding[:, 0::2, :] * w_rgcn_A.unsqueeze(1) + player_A_node_embedding * w_gcn_A.unsqueeze(1)
        tmp_embedding[:, 1::2, :] = full_graph_node_embedding[:, 1::2, :] * w_rgcn_B.unsqueeze(1) + player_B_node_embedding * w_gcn_B.unsqueeze(1)
        
        shot_type_predict = torch.cat((passed_node_embedding[:, :-2, :], tmp_embedding[:, -4:, :]), dim=1)
        
        if step % 2 == 0:
            black_node = shot_type_predict[:, (step - 1) * 2 + 1, :]
            white_node = shot_type_predict[:, (step - 2) * 2, :]
        if step % 2 == 1:
            black_node = shot_type_predict[:, (step - 2) * 2 + 1, :]
            white_node = shot_type_predict[:, (step - 1) * 2, :]

        type_predict_node = torch.cat((black_node, white_node), dim=-1) 
        predict_shot_type_logit = self.predict_shot_type(type_predict_node)
        # ==================================================================================================================================================================
        adjacency_matrix = update_adjacency_matrix(batch_size, step, adjacency_matrix)
        if train:
            for batch in range(batch_size):
                if step % 2 == 0:
                    adjacency_matrix[batch][shot_type[batch][0]][(step - 2) * 2][(step - 1) * 2 + 1] = 1
                    adjacency_matrix[batch][shot_type[batch][0]][(step - 1) * 2 + 1][(step - 2) * 2] = 1
                if step % 2 == 1:
                    adjacency_matrix[batch][shot_type[batch][0]][(step - 1) * 2][(step - 2) * 2 + 1] = 1
                    adjacency_matrix[batch][shot_type[batch][0]][(step - 2) * 2 + 1][(step - 1) * 2] = 1
        else:
            weights = predict_shot_type_logit[0, 1:]
            weights = F.softmax(weights, dim=0)
            predict_shot_type = torch.multinomial(weights, 1).unsqueeze(0) + 1

            for batch in range(batch_size):
                if step % 2 == 0:
                    adjacency_matrix[batch][predict_shot_type[batch][0]][(step - 2) * 2][(step - 1) * 2 + 1] = 1
                    adjacency_matrix[batch][predict_shot_type[batch][0]][(step - 1) * 2 + 1][(step - 2) * 2] = 1
                if step % 2 == 1:
                    adjacency_matrix[batch][predict_shot_type[batch][0]][(step - 1) * 2][(step - 2) * 2 + 1] = 1
                    adjacency_matrix[batch][predict_shot_type[batch][0]][(step - 2) * 2 + 1][(step - 1) * 2] = 1

        player_A_embedding = model_input[:, 0::2, :].clone()
        player_B_embedding = model_input[:, 1::2, :].clone()
        partial_adjacency_matrix = self.partial_adjacency_matrix[:player_A_embedding.size(1), :player_A_embedding.size(1)].to(player.device)
        partial_adjacency_matrix = torch.tile(partial_adjacency_matrix, (batch_size, 1, 1))

        dynamic_gcn_input_A = torch.cat((player_A_embedding, player_embedding[:, 0:1, :].repeat(1, player_A_embedding.size(1), 1)), dim=-1)
        dynamic_gcn_input_B = torch.cat((player_B_embedding, player_embedding[:, 1:2, :].repeat(1, player_B_embedding.size(1), 1)), dim=-1)

        dynamic_gcn_input_A = self.linear_for_dynmaic_gcn(dynamic_gcn_input_A)
        dynamic_gcn_input_B = self.linear_for_dynmaic_gcn(dynamic_gcn_input_B)

        player_A_node_embedding = self.gcn(dynamic_gcn_input_A, partial_adjacency_matrix)
        player_B_node_embedding = self.gcn(dynamic_gcn_input_B, partial_adjacency_matrix)
        
        full_graph_node_embedding = self.rGCN(model_input, adjacency_matrix)
        tmp_embedding = torch.zeros((full_graph_node_embedding.size(0), full_graph_node_embedding.size(1), full_graph_node_embedding.size(2))).to(player.device)

        _, _, A_weight, B_weight =  self.co_attention(player_A_node_embedding.permute(0, 2, 1), player_B_node_embedding, batch_size)
        A_weight = self.sigmoid(self.co_attention_linear_A(A_weight))
        B_weight = self.sigmoid(self.co_attention_linear_B(B_weight))

        player_A_node_embedding = player_A_node_embedding + B_weight.unsqueeze(1) * player_B_node_embedding
        player_B_node_embedding = player_B_node_embedding + A_weight.unsqueeze(1) * player_A_node_embedding

        rgcn_embedding_A = full_graph_node_embedding[:, 0::2, :].clone()[:, -1:, :].view(batch_size, -1)
        rgcn_embedding_B = full_graph_node_embedding[:, 1::2, :].clone()[:, -1:, :].view(batch_size, -1)
        gcn_embedding_A = player_A_node_embedding[:, -1:, :].clone().view(batch_size, -1)
        gcn_embedding_B = player_B_node_embedding[:, -1:, :].clone().view(batch_size, -1)

        rgcn_weight_A = self.rgcn_weight(rgcn_embedding_A)
        rgcn_weight_B = self.rgcn_weight(rgcn_embedding_B)
        gcn_weight_A = self.gcn_weight(gcn_embedding_A)
        gcn_weight_B = self.gcn_weight(gcn_embedding_B)
        
        w_rgcn_A = self.sigmoid(rgcn_weight_A)
        w_gcn_A = self.sigmoid(gcn_weight_A)
        w_rgcn_B = self.sigmoid(rgcn_weight_B)
        w_gcn_B = self.sigmoid(gcn_weight_B)

        tmp_embedding[:, 0::2, :] = full_graph_node_embedding[:, 0::2, :] * w_rgcn_A.unsqueeze(1) + player_A_node_embedding * w_gcn_A.unsqueeze(1)
        tmp_embedding[:, 1::2, :] = full_graph_node_embedding[:, 1::2, :] * w_rgcn_B.unsqueeze(1) + player_B_node_embedding * w_gcn_B.unsqueeze(1)

        last_two_node = tmp_embedding[:, -2:, :].view(batch_size, -1)
        predict_xy = self.predict_xy(last_two_node)             
        predict_xy = predict_xy.view(batch_size, 2, 5)
        
        return predict_xy, predict_shot_type_logit, adjacency_matrix, passed_node_embedding, original_embedding

class Encoder(nn.Module):
    def __init__(self, args, device):
        super(Encoder, self).__init__()
        player_num = args['player_num']
        player_dim = args['player_dim']

        type_num = args['type_num']

        location_dim = args['location_dim']

        hidden_size = args['hidden_size']

        num_layer = args['num_layer']

        self.player_num = player_num

        self.player_embedding = nn.Embedding(player_num, player_dim)
        self.coordination_transform = nn.Linear(2, location_dim)
        
        self.model_input_linear = nn.Linear(player_dim + location_dim , hidden_size)

        self.rGCN = relational_GCN(hidden_size, type_num, args['num_basis'], num_layer, device) # into 2 type (passive and active) and padding
        self.gcn = GCN(args['hidden_size'], args['hidden_size'], 0.1, num_layer, args, device)

        self.partial_adjacency_matrix = torch.ones((args['encode_length'], args['encode_length']), dtype=int) - torch.eye(args['encode_length'], dtype=int)

        self.rgcn_weight = nn.Linear(args['hidden_size'], 1)
        self.gcn_weight = nn.Linear(args['hidden_size'], 1)

        self.co_attention = ParallelCoAttentionNetwork(args['hidden_size'], args['hidden_size'], src_length_masking=False)
        self.co_attention_linear_A = nn.Linear(args['hidden_size'], 1)
        self.co_attention_linear_B = nn.Linear(args['hidden_size'], 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.linear_for_dynmaic_gcn = nn.Linear(args['hidden_size'] + args['player_dim'], args['hidden_size'])

    
    def forward(self, player, shot_type, player_A_x, player_A_y, player_B_x, player_B_y, encode_length):
        # get the initial(encode) adjacency matrix
        batch_size = player.size(0)
        adjacency_matrix = initialize_adjacency_matrix(batch_size, encode_length, shot_type)
        player_A_coordination = torch.cat((player_A_x.unsqueeze(2), player_A_y.unsqueeze(2)), dim=2).float()
        player_B_coordination = torch.cat((player_B_x.unsqueeze(2), player_B_y.unsqueeze(2)), dim=2).float()

        # interleave the player and opponent location
        coordination_sequence = torch.stack((player_A_coordination, player_B_coordination), dim=2).view(player.size(0), -1, 2)
        coordination_transform = self.coordination_transform(coordination_sequence)
        coordination_transform = F.relu(coordination_transform)

        player = player.repeat([1, encode_length])
        player_embedding = self.player_embedding(player)

        rally_information = torch.cat((coordination_transform, player_embedding), dim=-1)
        
        model_input = self.model_input_linear(rally_information)
        # fixed node embedding in decoder
        full_graph_node_embedding = self.rGCN( model_input, adjacency_matrix)
 
        player_A_embedding = model_input[:, 0::2, :].clone()
        player_B_embedding = model_input[:, 1::2, :].clone()

        partial_adjacency_matrix = torch.tile(self.partial_adjacency_matrix, (batch_size, 1, 1))

        dynamic_gcn_input_A = torch.cat((player_A_embedding, player_embedding[:, 0:1, :].repeat(1, encode_length, 1)), dim=-1)
        dynamic_gcn_input_B = torch.cat((player_B_embedding, player_embedding[:, 1:2, :].repeat(1, encode_length, 1)), dim=-1)

        dynamic_gcn_input_A = self.linear_for_dynmaic_gcn(dynamic_gcn_input_A)
        dynamic_gcn_input_B = self.linear_for_dynmaic_gcn(dynamic_gcn_input_B)

        player_A_node_embedding = self.gcn(dynamic_gcn_input_A, partial_adjacency_matrix)
        player_B_node_embedding = self.gcn(dynamic_gcn_input_B, partial_adjacency_matrix)
        node_embedding = torch.zeros((full_graph_node_embedding.size(0), full_graph_node_embedding.size(1), full_graph_node_embedding.size(2))).to(player.device)
        
        _, _, A_weight, B_weight = self.co_attention(player_A_node_embedding.permute(0, 2, 1), player_B_node_embedding, batch_size)
        
        A_weight = self.sigmoid(self.co_attention_linear_A(A_weight))
        B_weight = self.sigmoid(self.co_attention_linear_B(B_weight))

        player_A_node_embedding = player_A_node_embedding + B_weight.unsqueeze(1) * player_B_node_embedding
        player_B_node_embedding = player_B_node_embedding + A_weight.unsqueeze(1) * player_A_node_embedding

        rgcn_embedding_A = full_graph_node_embedding[:, 0::2, :].clone()[:, -1, :].view(batch_size, -1)
        rgcn_embedding_B = full_graph_node_embedding[:, 1::2, :].clone()[:, -1, :].view(batch_size, -1)
        gcn_embedding_A = player_A_node_embedding.clone()[:, -1, :].view(batch_size, -1)
        gcn_embedding_B = player_B_node_embedding.clone()[:, -1, :].view(batch_size, -1)

        rgcn_weight_A = self.rgcn_weight(rgcn_embedding_A)
        rgcn_weight_B = self.rgcn_weight(rgcn_embedding_B)
        gcn_weight_A = self.gcn_weight(gcn_embedding_A)
        gcn_weight_B = self.gcn_weight(gcn_embedding_B)
        
        w_rgcn_A = self.sigmoid(rgcn_weight_A)
        w_gcn_A = self.sigmoid(gcn_weight_A)
        w_rgcn_B = self.sigmoid(rgcn_weight_B)
        w_gcn_B = self.sigmoid(gcn_weight_B)

        node_embedding[:, 0::2, :] = full_graph_node_embedding[:, 0::2, :] * w_rgcn_A.unsqueeze(1) + player_A_node_embedding * w_gcn_A.unsqueeze(1)
        node_embedding[:, 1::2, :] = full_graph_node_embedding[:, 1::2, :] * w_rgcn_B.unsqueeze(1) + player_B_node_embedding * w_gcn_B.unsqueeze(1)

        return node_embedding, model_input, adjacency_matrix

# muti-head co attention 
