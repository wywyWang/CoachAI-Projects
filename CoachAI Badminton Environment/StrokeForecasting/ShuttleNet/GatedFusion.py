import torch.nn as nn
import torch
import torch.nn.functional as F


class GatedFusionLayer(nn.Module):
    def __init__(self, dim, encodeLength, n_position=200):
        super().__init__()
        n = 3
        self.linear_h_A = nn.Linear(dim, dim, bias=False)
        self.linear_h_A = nn.Linear(dim, dim, bias=False)
        self.linear_h_A = nn.Linear(dim, dim, bias=False)
        self.linear_alpha_A = nn.Linear(dim * n, dim, bias=False)
        self.linear_alpha_B = nn.Linear(dim * n, dim, bias=False)
        self.linear_alpha_L = nn.Linear(dim * n, dim, bias=False)

        self.decode_length = n_position - encodeLength

        self.w_A = nn.Parameter(torch.zeros([self.decode_length, dim]), requires_grad=True)
        self.w_B = nn.Parameter(torch.zeros([self.decode_length, dim]), requires_grad=True)
        self.w_L = nn.Parameter(torch.zeros([self.decode_length, dim]), requires_grad=True)
    
    def project(self, x_A, x_B, x_L):
        h_A = torch.tanh(self.linear_h_A(x_A))
        h_B = torch.tanh(self.linear_h_A(x_B))
        h_L = torch.tanh(self.linear_h_A(x_L))

        return h_A, h_B, h_L

    def forward(self, x_A, x_B, x_L):
        # (batch, seq, dim)
        batch, seq_len, dim = x_A.shape
        w_A = self.w_A.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B = self.w_B.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L = self.w_L.unsqueeze(0).repeat_interleave(batch, dim=0) 

        beta_A = w_A[:, :seq_len, :]
        beta_B = w_B[:, :seq_len, :]
        beta_L = w_L[:, :seq_len, :]

        x = torch.cat((x_A, x_B, x_L), dim=-1)
        alpha_A = torch.sigmoid(self.linear_alpha_A(x))
        alpha_B = torch.sigmoid(self.linear_alpha_B(x))
        alpha_L = torch.sigmoid(self.linear_alpha_L(x))

        h_A, h_B, h_L = self.project(x_A, x_B, x_L)

        output_A = beta_A * alpha_A * h_A
        output_B = beta_B * alpha_B * h_B
        output_L = beta_L * alpha_L * h_L

        return torch.sigmoid(output_A + output_B + output_L)