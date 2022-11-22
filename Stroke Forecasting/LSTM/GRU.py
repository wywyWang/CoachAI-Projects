import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os


PAD = 0


class PlayerEmbedding(nn.Embedding):
    def __init__(self, player_num, embed_dim):
        super(PlayerEmbedding, self).__init__(player_num, embed_dim, padding_idx=PAD)


class ShotEmbedding(nn.Embedding):
    def __init__(self, shot_num, embed_dim):
        super(ShotEmbedding, self).__init__(shot_num, embed_dim, padding_idx=PAD)


class Predictor(nn.Module):
    def __init__(self, config):
        super(Predictor, self).__init__()
        self.area_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['area_num'], bias=False)
        )
        self.shot_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['shot_num'], bias=False)
        )
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])


    def forward(self, x, target_player):
        embedded_player = self.player_embedding(target_player)
        x = (x + embedded_player)

        area_logits = self.area_decoder(x)
        shot_logits = self.shot_decoder(x)

        return area_logits, shot_logits


class GRUDecoder(nn.Module):
    def __init__(self, config, n_layers=1):
        super(GRUDecoder, self).__init__()
        self.scale_emb = False
        self.d_model = config['encode_dim']
        self.n_layers = n_layers
        self.hidden_size = config['encode_dim']

        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])
        self.linear = nn.Linear(config['shot_dim']+config['area_dim'], config['player_dim'])
        
        self.gru_decoder = nn.LSTM(input_size=self.hidden_size, hidden_size=config['encode_dim'], num_layers=config['n_layers'], batch_first=True, bidirectional=False)
        self.linear2 = nn.Linear(config['encode_dim']*config['num_directions'], config['encode_dim'])
        self.predictor = Predictor(config)

    def forward(self, input_shot, input_x, input_y, input_player, target_player, hidden, cell):
        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()

        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)

        embedded_area += embedded_player
        embedded_shot += embedded_player

        # add player embedding to shot and area embeddings
        embedded_shot_area = torch.cat((embedded_shot, embedded_area), dim=-1)       # (batch, seq_len, embed_dim)
        embedded_shot_area = self.linear(embedded_shot_area)

        # Forward
        if self.scale_emb:
            embedded_shot_area *= self.d_model ** 0.5

        output, (hidden, cell) = self.gru_decoder(embedded_shot_area, (hidden, cell))
        output = self.linear2(output)

        output_area_logits, output_shot_logits = self.predictor(output, target_player)

        return output_area_logits, output_shot_logits, hidden, cell


class GRUEncoder(nn.Module):
    def __init__(self, config, n_layers=1):
        super(GRUEncoder, self).__init__()
        self.scale_emb = False
        self.d_model = config['encode_dim']
        self.n_layers = n_layers
        self.hidden_size = config['encode_dim']

        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])
        self.linear = nn.Linear(config['shot_dim']+config['area_dim'], config['player_dim'])

        self.gru_encoder = nn.LSTM(input_size=self.hidden_size, hidden_size=config['encode_dim'], num_layers=config['n_layers'], batch_first=True, bidirectional=False)

    def forward(self, input_shot, input_x, input_y, input_player, hidden, cell):
        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()

        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)

        # add player embedding to shot and area embeddings
        embedded_area += embedded_player
        embedded_shot += embedded_player

        embedded_shot_area = torch.cat((embedded_shot, embedded_area), dim=-1)       # (batch, seq_len, embed_dim)
        embedded_shot_area = self.linear(embedded_shot_area)

        # Forward
        if self.scale_emb:
            embedded_shot_area *= self.d_model ** 0.5

        output, (hidden, cell) = self.gru_encoder(embedded_shot_area, (hidden, cell))                        # (batch, seq_len, embed_dim) -> (batch, seq_len, n_directions * hidden_size)

        return output, hidden, cell