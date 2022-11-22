import torch.nn as nn
import torch
from Transformer.transformer_submodules import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, encode_input, slf_attn_mask=None):
        encode_output, enc_slf_attn = self.slf_attn(encode_input, encode_input, encode_input, mask=slf_attn_mask)
        encode_output = self.pos_ffn(encode_output)
        return encode_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.encode_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, decode_input, encode_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        decode_output, dec_slf_attn = self.slf_attn(decode_input, decode_input, decode_input, mask=slf_attn_mask)
        decode_output, dec_enc_attn = self.encode_attn(decode_output, encode_output, encode_output, mask=dec_enc_attn_mask)
        decode_output = self.pos_ffn(decode_output)
        return decode_output, dec_slf_attn, dec_enc_attn