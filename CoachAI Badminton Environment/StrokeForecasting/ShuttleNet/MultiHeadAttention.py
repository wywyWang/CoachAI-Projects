import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, attn_dropout : float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        # Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
        d_k = q.size(-1)
                                              #transpose last two dim
        attn = torch.matmul(q / np.sqrt(d_k), k.transpose(2, 3))

        # for decoder, we need to mask out the msg after current
        if mask is not None:
            # mask fill with -∞
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, headCount : int, model_dim : int, dropout : float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim
        self.headCount = headCount
        self.linear_k = nn.Linear(model_dim, model_dim * headCount, bias = False)
        self.linear_v = nn.Linear(model_dim, model_dim * headCount, bias = False)
        self.linear_q = nn.Linear(model_dim, model_dim * headCount, bias = False)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim * headCount, model_dim, bias = False)
        self.dropout = nn.Dropout(dropout)
	
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)

    def forward(self, q : torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        residual = q

        dim_per_head = self.dim_per_head
        headCount = self.headCount

        # q_dim: 30 * 16 * 32
        dataCount = q.size(0)
        len_q = q.size(1)

        # linear projection then split by heads
        k = self.linear_k(k).view(dataCount, headCount, -1, dim_per_head)
        v = self.linear_v(v).view(dataCount, headCount, -1, dim_per_head)
        q = self.linear_q(q).view(dataCount, headCount, -1, dim_per_head)

        if mask is not None:
            mask = mask.unsqueeze(1)

        # scaled dot product attention
        context = self.dot_product_attention(q, k, v, mask)

        # concat heads
        context = context.transpose(1, 2).contiguous().view(dataCount, len_q, -1)

        # final linear projection then dropout
        output = self.dropout(self.linear_final(context))

        # add residual
        output += residual

        # add residual and norm layer
        output = self.layer_norm(output)

        return output


class TypeAreaScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention of type-area attention'''

    def __init__(self, scale, attn_dropout=0.1):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q_a : torch.Tensor, k_a : torch.Tensor, v_a : torch.Tensor, 
                      q_s : torch.Tensor, k_s : torch.Tensor, v_s : torch.Tensor, mask : torch.Tensor=None) -> torch.Tensor:
        a2a = torch.matmul(q_a, k_a.transpose(2, 3))
        a2s = torch.matmul(q_a, k_s.transpose(2, 3))
        s2a = torch.matmul(q_s, k_a.transpose(2, 3))
        s2s = torch.matmul(q_s, k_s.transpose(2, 3))
        attention_score = (a2a + a2s + s2a + s2s) / self.scale
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        attention_score = self.dropout(F.softmax(attention_score, dim=-1))
        output = torch.matmul(attention_score, (v_a + v_s))

        return output


class TypeAreaMultiHeadAttention(nn.Module):
    ''' Multi-Head Type-Area Attention module '''

    def __init__(self, n_head : int, d_model : int, dropout : float=0.1):
        super().__init__()

        self.dim_per_head = d_model
        self.n_head = n_head

        self.linear_qs = nn.Linear(d_model, n_head * d_model, bias=False)
        self.linear_ks = nn.Linear(d_model, n_head * d_model, bias=False)
        self.linear_vs = nn.Linear(d_model, n_head * d_model, bias=False)

        self.linear_qa = nn.Linear(d_model, n_head * d_model, bias=False)
        self.linear_ka = nn.Linear(d_model, n_head * d_model, bias=False)
        self.linear_va = nn.Linear(d_model, n_head * d_model, bias=False)

        self.linear_final = nn.Linear(n_head * d_model, d_model, bias=False)

        scaling_factor = (4 * d_model)**0.5

        self.attention = TypeAreaScaledDotProductAttention(scale=scaling_factor)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q_a : torch.Tensor, k_a : torch.Tensor, v_a : torch.Tensor,
                      q_s : torch.Tensor, k_s : torch.Tensor, v_s : torch.Tensor, mask : torch.Tensor=None) -> torch.Tensor:
        n_head = self.n_head
        dim_per_head = self.dim_per_head
        sz_b, len_q, len_k, len_v = q_a.size(0), q_a.size(1), k_a.size(1), v_a.size(1)

        residual_a = q_a
        residual_s = q_s

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q_a = self.linear_qa(q_a).view(sz_b, n_head, len_q, dim_per_head)
        k_a = self.linear_ka(k_a).view(sz_b, n_head, len_k, dim_per_head)
        v_a = self.linear_va(v_a).view(sz_b, n_head, len_v, dim_per_head)

        q_s = self.linear_qs(q_s).view(sz_b, n_head, len_q, dim_per_head)
        k_s = self.linear_ks(k_s).view(sz_b, n_head, len_k, dim_per_head)
        v_s = self.linear_vs(v_s).view(sz_b, n_head, len_v, dim_per_head)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        output = self.attention(q_a, k_a, v_a, q_s, k_s, v_s, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.linear_final(output))

        #add & normalize
        output += residual_a + residual_s
        output = self.layer_norm(output)

        return output