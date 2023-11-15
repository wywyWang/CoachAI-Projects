import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)


    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)

        # Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
        attn = torch.matmul(q / np.sqrt(d_k), k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    #d_model = d_k = d_v
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc   = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q


class TypeAreaScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention of type-area attention'''

    def __init__(self, scale, attn_dropout=0.1):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q_a, k_a, v_a, q_s, k_s, v_s, mask=None):
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

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.w_qa = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ka = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_va = nn.Linear(d_model, n_head * d_v, bias=False)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        scaling_factor = (4 * d_k) ** 0.5

        self.attention = TypeAreaScaledDotProductAttention(scale=scaling_factor)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q_a, k_a, v_a, q_s, k_s, v_s, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q_a.size(0), q_a.size(1), k_a.size(1), v_a.size(1)

        residual_a = q_a
        residual_s = q_s

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q_a = self.w_qa(q_a).view(sz_b, len_q, n_head, d_k)
        k_a = self.w_ka(k_a).view(sz_b, len_k, n_head, d_k)
        v_a = self.w_va(v_a).view(sz_b, len_v, n_head, d_v)

        q_s = self.w_qs(q_s).view(sz_b, len_q, n_head, d_k)
        k_s = self.w_ks(k_s).view(sz_b, len_k, n_head, d_k)
        v_s = self.w_vs(v_s).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q_a, k_a, v_a = q_a.transpose(1, 2), k_a.transpose(1, 2), v_a.transpose(1, 2)
        q_s, k_s, v_s = q_s.transpose(1, 2), k_s.transpose(1, 2), v_s.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        output = self.attention(q_a, k_a, v_a, q_s, k_s, v_s, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))

        #add & normalize
        output += residual_a + residual_s
        output = self.layer_norm(output)

        return output