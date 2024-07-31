import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Norm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.size = emb_dim
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True, unbiased=False) + self.eps) + self.bias
        norm.retain_grad()
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0 , -1e9)

    scores = torch.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, emb_dim, dropout = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_k = emb_dim // heads
        self.h = heads
        
        self.q_linear = nn.Linear(emb_dim, emb_dim)
        self.v_linear = nn.Linear(emb_dim, emb_dim)
        self.k_linear = nn.Linear(emb_dim, emb_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(emb_dim, emb_dim)
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * emb_dim
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.emb_dim)
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, emb_dim, d_ff=1024, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 1024
        self.linear_1 = nn.Linear(emb_dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, emb_dim)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
