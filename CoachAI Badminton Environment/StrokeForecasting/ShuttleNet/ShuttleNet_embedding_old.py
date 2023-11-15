import torch
import torch.nn as nn
import numpy as np


PADDING = 0


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, encodeLen, n_position=200):
        super().__init__()

        self.encodingLen = encodeLen

        self.register_buffer('pos_table', 
                self.get_sinusoid_encoding_table(n_position, d_hid))

    def get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        table = np.array([get_position_angle_vec(pos_i)
                                  for pos_i in range(n_position)])
        table[:, 0::2] = np.sin(table[:, 0::2])  # dim 2i
        table[:, 1::2] = np.cos(table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(table).unsqueeze(0)

    def forward(self, x, mode='encode'):
        if mode == 'encode':
            return x + self.pos_table[:, :x.size(1)].clone().detach()
        elif mode == 'decode':
            # need to offset when decoding
            return x + self.pos_table[:, self.encodingLen:self.encodingLen+x.size(1)].clone().detach()

class PlayerEmbedding(nn.Embedding):
    def __init__(self, player_num, embed_dim):  # player_num is player count + 1(padding), ex. two player => 1,2 means player, 0 means padding
        super().__init__(player_num, embed_dim, padding_idx=PADDING)


class ShotEmbedding(nn.Embedding):
    def __init__(self, shot_num, embed_dim):  # shot_num is the type count of shot + 1(padding)
        super().__init__(shot_num, embed_dim, padding_idx=PADDING)
