import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

PAD = 0
# make adjacency matrix according to encode length which number of row is encode length times 2
def initialize_adjacency_matrix(batch_size, encode_length, shot_type):
    adjacency_matrix = torch.ones(((encode_length) * 2, (encode_length) * 2), dtype=int).to(shot_type.device)
    adjacency_matrix = adjacency_matrix - torch.eye((encode_length) * 2).to(shot_type.device)
    adjacency_matrix = torch.tile(adjacency_matrix, (batch_size, 1, 1))

    return adjacency_matrix

def update_adjacency_matrix(batch_size, step, adjacency_matrix, shot_type_predict=False):
    new_adjacency_matrix = torch.ones(((step) * 2, (step) * 2), dtype=int).to(adjacency_matrix.device)
    adjacency_matrix = new_adjacency_matrix - torch.eye((step) * 2).to(adjacency_matrix.device)
    adjacency_matrix = torch.tile(adjacency_matrix, (batch_size, 1, 1))

    if shot_type_predict:        
        if step % 2 == 0:
            adjacency_matrix[:, (step - 1) * 2, :] = 0
            adjacency_matrix[:, :, (step - 1) * 2] = 0
        if step % 2 == 1: 
            adjacency_matrix[:, (step - 1) * 2 + 1, :] = 0
            adjacency_matrix[:, :, (step - 1) * 2 + 1] = 0

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

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_embedding, adjacency_matrix, activation_function):
        
        node_embedding = torch.matmul(adjacency_matrix, node_embedding)

        output = self.linear(node_embedding)
        output = activation_function(output)

        output = self.dropout(output)

        return output

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, num_layer):
        super(GCN, self).__init__()
        self.gcn_layer_list = nn.ModuleList([GCNLayer(input_dim, hidden_dim, dropout) for _ in range(num_layer)])
        
        self.hidden_activation_function = nn.ReLU()
        self.output_activation_function = nn.Sigmoid()
        
    def forward(self, node_embedding, adjacency_matrix):
        adjacency_matrix = preprocess_adj(adjacency_matrix).float().to(node_embedding.device)

        for i, gcn_layer in enumerate(self.gcn_layer_list):
            if i == len(self.gcn_layer_list) - 1:
                node_embedding = gcn_layer(node_embedding, adjacency_matrix, self.output_activation_function)
            else:
                node_embedding = gcn_layer(node_embedding, adjacency_matrix, self.hidden_activation_function)
        
        return node_embedding

class Decoder(nn.Module):
    def __init__(self, args):
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

        self.gcn = GCN(hidden_size, hidden_size, args['dropout'], num_layer)

        self.predict_shot_type = nn.Linear(hidden_size * 2, type_num)
        self.predict_xy = nn.Linear(hidden_size * 2, 10)
 
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
            tmp_embedding = self.gcn(model_input, adjacency_matrix)
            passed_node_embedding = torch.cat((encode_node_embedding[:, :-2, :], tmp_embedding[:, -4:, :]), dim=1)
        else:
            passed_node_embedding = encode_node_embedding.clone()
     
        batch_size = player.size(0)
        tmp_adjacency_matrix = update_adjacency_matrix(batch_size, step, adjacency_matrix, shot_type_predict=True)
        model_input = torch.cat((passed_node_embedding, initial_embedding), dim=1)
        # ========================================================================================================================
        if step % 2 == 0:            
            tmp_model_input = model_input[:, torch.arange(model_input.size(1))!=(step-1)*2]
            tmp_adjacency_matrix = tmp_adjacency_matrix[:, torch.arange(tmp_adjacency_matrix.size(1))!=(step-1)*2]
            tmp_adjacency_matrix = tmp_adjacency_matrix[:, :, torch.arange(tmp_adjacency_matrix.size(2))!=(step-1)*2]
        if step % 2 == 1:
            tmp_model_input = model_input[:, torch.arange(model_input.size(1))!=(step-1)*2+1]
            tmp_adjacency_matrix = tmp_adjacency_matrix[:, torch.arange(tmp_adjacency_matrix.size(1))!=(step-1)*2+1]
            tmp_adjacency_matrix = tmp_adjacency_matrix[:, :, torch.arange(tmp_adjacency_matrix.size(2))!=(step-1)*2+1]

        tmp_embedding = self.gcn(tmp_model_input, tmp_adjacency_matrix)
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
        #=================================================================================================================================
        adjacency_matrix = update_adjacency_matrix(batch_size, step, adjacency_matrix)

        if train:
            for batch in range(batch_size):
                if step % 2 == 0:
                    adjacency_matrix[batch][(step - 2) * 2][(step - 1) * 2 + 1] = 1
                    adjacency_matrix[batch][(step - 1) * 2 + 1][(step - 2) * 2] = 1
                if step % 2 == 1:
                    adjacency_matrix[batch][(step - 1) * 2][(step - 2) * 2 + 1] = 1
                    adjacency_matrix[batch][(step - 2) * 2 + 1][(step - 1) * 2] = 1
        else:
            for batch in range(batch_size):
                if step % 2 == 0:
                    adjacency_matrix[batch][(step - 2) * 2][(step - 1) * 2 + 1] = 1
                    adjacency_matrix[batch][(step - 1) * 2 + 1][(step - 2) * 2] = 1
                if step % 2 == 1:
                    adjacency_matrix[batch][(step - 1) * 2][(step - 2) * 2 + 1] = 1
                    adjacency_matrix[batch][(step - 2) * 2 + 1][(step - 1) * 2] = 1
            
        tmp_embedding = self.gcn(model_input, adjacency_matrix)
        node_embedding = torch.cat((model_input[:, :-2, :], tmp_embedding[:, -2:, :]), dim=1)

        last_two_node = node_embedding[:, -2:, :].view(batch_size, -1)
        predict_xy = self.predict_xy(last_two_node)             
        predict_xy = predict_xy.view(batch_size, 2, 5)

        return predict_xy, predict_shot_type_logit, adjacency_matrix, passed_node_embedding

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        player_num = args['player_num']
        player_dim = args['player_dim']

        location_dim = args['location_dim']

        hidden_size = args['hidden_size']

        self.player_num = player_num
        
        num_layer = args['num_layer']

        self.player_embedding = nn.Embedding(player_num, player_dim)
        self.coordination_transform = nn.Linear(2, location_dim)
        
        self.model_input_linear = nn.Linear(player_dim + location_dim , hidden_size)

        self.gcn = GCN(hidden_size, hidden_size, 0.1, num_layer)

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

        # player = player + self.player_num * set
        player = player.repeat([1, encode_length])
        player_embedding = self.player_embedding(player)

        rally_information = torch.cat((coordination_transform, player_embedding), dim=-1)
        
        model_input = self.model_input_linear(rally_information)
        # fixed node embedding in decoder
        node_embedding = self.gcn(model_input, adjacency_matrix)
        
        return node_embedding, adjacency_matrix
