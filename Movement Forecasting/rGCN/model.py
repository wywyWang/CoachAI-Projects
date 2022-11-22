import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

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

        self.player_num = args['player_num']
        self.type_num = args['type_num']

        num_layer = args['num_layer']

        self.player_embedding = nn.Embedding(player_num, player_dim)
        self.coordination_transform = nn.Linear(2, location_dim)

        self.model_input_linear = nn.Linear(player_dim + location_dim, hidden_size)

        self.rGCN = relational_GCN(hidden_size, type_num, args['num_basis'], num_layer, device)

        self.predict_shot_type = nn.Linear(hidden_size * 2, type_num)
        self.predict_xy = nn.Linear(hidden_size*2, 10)

    def forward(self, player, step, encode_node_embedding, adjacency_matrix, 
                player_A_x, player_A_y, player_B_x, player_B_y, 
                shot_type=None, train=False, first=False):
        batch_size = player.size(0)

        prev_player_A_coordination = torch.cat((player_A_x.unsqueeze(2), player_A_y.unsqueeze(2)), dim=2).float()
        prev_player_B_coordination = torch.cat((player_B_x.unsqueeze(2), player_B_y.unsqueeze(2)), dim=2).float()

        player_embedding = self.player_embedding(player)

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
            model_input = torch.cat((encode_node_embedding, model_input), dim=1)
            tmp_embedding = self.rGCN(model_input, adjacency_matrix)
            passed_node_embedding = torch.cat((encode_node_embedding[:, :-2, :], tmp_embedding[:, -4:, :]), dim=1)
        else:
            passed_node_embedding = encode_node_embedding.clone()
     
        batch_size = player.size(0)
        tmp_adjacency_matrix = update_adjacency_matrix(batch_size, step, adjacency_matrix, shot_type_predict=True)
        model_input = torch.cat((passed_node_embedding, initial_embedding), dim=1)
        # ===============================================================================================================
        if step % 2 == 0:            
            tmp_model_input = model_input[:, torch.arange(model_input.size(1))!=(step-1)*2]
            tmp_adjacency_matrix = tmp_adjacency_matrix[:, :, torch.arange(tmp_adjacency_matrix.size(2))!=(step-1)*2]
            tmp_adjacency_matrix = tmp_adjacency_matrix[:, :, :, torch.arange(tmp_adjacency_matrix.size(3))!=(step-1)*2]
        if step % 2 == 1:
            tmp_model_input = model_input[:, torch.arange(model_input.size(1))!=(step-1)*2+1]
            tmp_adjacency_matrix = tmp_adjacency_matrix[:, :, torch.arange(tmp_adjacency_matrix.size(2))!=(step-1)*2+1]
            tmp_adjacency_matrix = tmp_adjacency_matrix[:, :, :, torch.arange(tmp_adjacency_matrix.size(3))!=(step-1)*2+1]

        tmp_embedding = self.rGCN(tmp_model_input, tmp_adjacency_matrix)
        padding_full_graph_node_embedding = torch.zeros((tmp_embedding.size(0), tmp_embedding.size(1)+1, tmp_embedding.size(2))).to(player.device)

        if step % 2 == 0:
            padding_full_graph_node_embedding[:, :-2, :] = tmp_embedding[:, :-1, :]
            padding_full_graph_node_embedding[:, -1, :] = tmp_embedding[:, -1, :]  
        if step % 2 == 1: 
            padding_full_graph_node_embedding[:, :-1, :] = tmp_embedding

        shot_type_predict = torch.cat((passed_node_embedding[:, :-2, :], padding_full_graph_node_embedding[:, -4:, :]), dim=1)
        
        if step % 2 == 0:
            black_node = shot_type_predict[:, (step - 1) * 2 + 1, :]
            white_node = shot_type_predict[:, (step - 2) * 2, :]
        if step % 2 == 1:
            black_node = shot_type_predict[:, (step - 2) * 2 + 1, :]
            white_node = shot_type_predict[:, (step - 1) * 2, :]            

        type_predict_node = torch.cat((black_node, white_node), dim=-1) 
        predict_shot_type_logit = self.predict_shot_type(type_predict_node)

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

            
        tmp_embedding = self.rGCN(model_input, adjacency_matrix)
        node_embedding = torch.cat((model_input[:, :-2, :], tmp_embedding[:, -2:, :]), dim=1)

        last_two_node = node_embedding[:, -2:, :].view(batch_size, -1)
        predict_xy = self.predict_xy(last_two_node)             
        predict_xy = predict_xy.view(batch_size, 2, 5)
        
        return predict_xy, predict_shot_type_logit, adjacency_matrix, passed_node_embedding

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
        node_embedding = self.rGCN( model_input, adjacency_matrix)
        
        return node_embedding, adjacency_matrix