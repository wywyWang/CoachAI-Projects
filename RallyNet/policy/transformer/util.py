import torch
import torch.nn as nn
import math
import numpy as np
from torch.autograd import Variable

class PositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_seq_len = 20, dropout = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, emb_dim)
        for pos in range(max_seq_len):
            for i in range(0, emb_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/emb_dim)))
                if (i + 1) < emb_dim:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/emb_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x): # input_dim: batch_size x seq_len x emb_dim
        # make embeddings relatively larger
        x = x * math.sqrt(self.emb_dim)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
            x.cuda()
        else:
            pe = pe.cpu()
            x.cpu()
        x = x + pe
        return self.dropout(x)