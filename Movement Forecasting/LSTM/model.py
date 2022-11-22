import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 0

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        player_num = args['player_num']
        player_dim = args['player_dim']

        type_num = args['type_num']
        type_dim = args['type_dim']

        location_dim = args['location_dim']

        hidden_size = args['hidden_size']

        self.player_embedding = nn.Embedding(player_num, player_dim)
        self.type_embedding = nn.Embedding(type_num, type_dim)
        self.self_coordination_transform = nn.Linear(2, location_dim)
        self.coordination_transform = nn.Linear(location_dim*2, location_dim)

        self.location_player = nn.Linear(location_dim+player_dim, hidden_size)
        self.type_player = nn.Linear(type_dim+player_dim, hidden_size)

        self.model_input_linear = nn.Linear(location_dim + type_dim, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)

        self.predict_xy = nn.Linear(hidden_size, 10)

        self.predict_shot_type = nn.Linear(hidden_size, type_num)

    def forward(self, player, player_A_x, player_A_y, player_B_x, player_B_y, shot_type, hidden, cell, two_player=None):
        player_embedding = self.player_embedding(player)
        two_player = self.player_embedding(two_player)

        player_A_coordination = torch.cat((player_A_x.unsqueeze(2), player_A_y.unsqueeze(2)), dim=2).float()
        player_B_coordination = torch.cat((player_B_x.unsqueeze(2), player_B_y.unsqueeze(2)), dim=2).float()
        
        player_A_coordination = self.self_coordination_transform(player_A_coordination)
        player_B_coordination = self.self_coordination_transform(player_B_coordination)
        player_A_coordination = F.relu(player_A_coordination)
        player_B_coordination = F.relu(player_B_coordination)            
        
        player_A_coordination = torch.cat((player_A_coordination, two_player[:, 0:1, :].repeat(1, player.size(1), 1)), dim=-1)
        player_B_coordination = torch.cat((player_B_coordination, two_player[:, 1:2, :].repeat(1, player.size(1), 1)), dim=-1)

        player_A_coordination = self.location_player(player_A_coordination)
        player_B_coordination = self.location_player(player_B_coordination)
        # interleave the player and opponent location
        coordination_sequence = torch.cat((player_A_coordination, player_B_coordination), dim=2)
        coordination_transform = self.coordination_transform(coordination_sequence)

        type_embedding = self.type_embedding(shot_type)
        type_embedding = torch.cat((type_embedding, player_embedding), dim=-1)
        type_embedding = self.type_player(type_embedding)

        model_input = torch.cat((coordination_transform, type_embedding), dim=-1)    
        model_input = self.model_input_linear(model_input) 

        output, (hidden, cell) = self.lstm(model_input, (hidden, cell))

        predict_xy = self.predict_xy(output)
 
        predict_shot_type_logit = self.predict_shot_type(output)
        
        return predict_xy, predict_shot_type_logit, hidden, cell

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        player_num = args['player_num']
        player_dim = args['player_dim']

        type_num = args['type_num']

        location_dim = args['location_dim']

        hidden_size = args['hidden_size']
        type_dim = args['type_dim']

        self.player_embedding = nn.Embedding(player_num, player_dim)
        self.type_embedding = nn.Embedding(type_num, type_dim)

        self.self_coordination_transform = nn.Linear(2, location_dim)
        self.coordination_transform = nn.Linear(location_dim*2, location_dim)

        self.location_player = nn.Linear(location_dim+player_dim, hidden_size)
        self.type_player = nn.Linear(type_dim+player_dim, hidden_size)

        self.model_input_linear = nn.Linear(location_dim + type_dim, hidden_size)       

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
    
    def forward(self, player, player_A_x, player_A_y, player_B_x, player_B_y, shot_type, hidden, cell, two_player=None):
        player_embedding = self.player_embedding(player)
        two_player = self.player_embedding(two_player)

        player_A_coordination = torch.cat((player_A_x.unsqueeze(2), player_A_y.unsqueeze(2)), dim=2).float()
        player_B_coordination = torch.cat((player_B_x.unsqueeze(2), player_B_y.unsqueeze(2)), dim=2).float()
        player_A_coordination = self.self_coordination_transform(player_A_coordination)
        player_B_coordination = self.self_coordination_transform(player_B_coordination)
        player_A_coordination = F.relu(player_A_coordination)
        player_B_coordination = F.relu(player_B_coordination)

        player_A_coordination = torch.cat((player_A_coordination, two_player[:, 0:1, :].repeat(1, player.size(1), 1)), dim=-1)
        player_B_coordination = torch.cat((player_B_coordination, two_player[:, 1:2, :].repeat(1, player.size(1), 1)), dim=-1)

        player_A_coordination = self.location_player(player_A_coordination)
        player_B_coordination = self.location_player(player_B_coordination)

        coordination_sequence = torch.cat((player_A_coordination, player_B_coordination), dim=2)
        coordination_transform = self.coordination_transform(coordination_sequence)

        type_embedding = self.type_embedding(shot_type)
        type_embedding = torch.cat((type_embedding, player_embedding), dim=-1)
        type_embedding = self.type_player(type_embedding)

        model_input = torch.cat((coordination_transform, type_embedding), dim=-1)
        model_input = self.model_input_linear(model_input)

        _, (hidden, cell) = self.lstm(model_input, (hidden, cell))

        return hidden, cell