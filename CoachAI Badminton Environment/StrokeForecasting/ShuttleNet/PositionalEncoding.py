import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, encodeLength: int, n_position: int = 200):
        super().__init__()

        self.encodingLength = encodeLength

        position = torch.arange(n_position).unsqueeze(1)    # dim = (n_position, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        table = torch.zeros(1, n_position, d_model)
        table[0, :, 0::2] = torch.sin(position * div_term)
        table[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_table', table)

    def forward(self, x: torch.Tensor, mode : str ='encode') -> torch.Tensor:
        if mode == 'encode':
            return x + self.pos_table[:, :x.size(1)].clone().detach()
        elif mode == 'decode':
            # need to offset when decoding
            return x + self.pos_table[:, self.encodingLength:self.encodingLength+x.size(1)].clone().detach()
