import torch
import torch.nn as nn
import torch.nn.functional as F
#from ShuttleNet.ShuttleNet_submodules import TypeAreaMultiHeadAttention#, PositionwiseFeedForward
from ShuttleNet.PositionwiseFeedForward import PositionwiseFeedForward
from ShuttleNet.MultiHeadAttention import TypeAreaMultiHeadAttention
from ShuttleNet.PositionalEncoding import PositionalEncoding
from ShuttleNet.Embedding import EmbeddingLayer

PAD = 0

class EncodeModule(nn.Module):
    #d_model = d_k = d_v = d_inner/2
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, config, dropout=0.1):
        super().__init__()
        self.positionalEncoding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round'])
        self.dropout = nn.Dropout(p=dropout)
        self.multiHeadAttention = TypeAreaMultiHeadAttention(n_head, d_model, dropout=dropout)
        self.FFN = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, h_a, h_s, mask=None):
        area = self.dropout(self.positionalEncoding(h_a, mode='encode'))
        shot = self.dropout(self.positionalEncoding(h_s, mode='encode'))

        # Multi-head type-area attention
        output = self.multiHeadAttention(area, area, area, shot, shot, shot, mask=mask)
        # FFN & add + normal
        output = self.FFN(output)
        return output


class ShotGenEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1

        self.embedding = EmbeddingLayer(config['area_dim'], config['shot_dim'], config['player_dim'], config['shot_num'],config['player_num'])

        self.playerEncode = EncodeModule(d_model, d_inner, n_heads, d_k, d_v, config =config, dropout=dropout)
        self.rallyEncode = EncodeModule(d_model, d_inner, n_heads, d_k, d_v, config =config, dropout=dropout)

    def forward(self, shot, x, y, player, src_mask=None):
        '''
        shot   = (dataCount, encodeLength) each shot is int and [1, 10]
        x      = (dataCount, encodeLength) each y is standardlize coord
        y      = (dataCount, encodeLength) each y is standardlize coord
        player = (dataCount, encodeLength) each player is {1,2}
        '''
        # area = (dataCount, encodeLength, (x, y))
        area = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), dim=-1).float()

        embedded_area, embedded_shot, embedded_player = self.embedding(area, shot, player)

        h_a = embedded_area + embedded_player
        h_s = embedded_shot + embedded_player

        # split player
        h_a_A = h_a[:, ::2]     #even
        h_a_B = h_a[:, 1::2]    #odd
        h_s_A = h_s[:, ::2]     #even
        h_s_B = h_s[:, 1::2]    #odd

        #multi-head type-area attention & FFN
        #feed data for one player
        # player A
        playerA = self.playerEncode(h_a_A, h_s_A, mask=src_mask)
        # player B
        playerB = self.playerEncode(h_a_B, h_s_B, mask=src_mask)
        # rally
        rally = self.rallyEncode(h_a, h_s, mask=src_mask)


        return rally, playerA, playerB