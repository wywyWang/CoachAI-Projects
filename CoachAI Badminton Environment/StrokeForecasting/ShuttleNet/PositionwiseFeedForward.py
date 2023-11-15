import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_in, d_hid)
        self.linear_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.linear_1(x)
        x = self.linear_2(F.gelu(x))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x