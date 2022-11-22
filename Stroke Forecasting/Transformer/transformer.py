import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import pickle5
from Transformer.transformer_submodules import MultiHeadAttention, PositionwiseFeedForward
from Transformer.transformer_layers import EncoderLayer, DecoderLayer
from Transformer.transformer_decoder import PositionalEncoding


PAD = 0


def get_pad_mask(seq, PAD):
    return (seq != PAD).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PlayerEmbedding(nn.Embedding):
    def __init__(self, player_num, embed_dim):
        super(PlayerEmbedding, self).__init__(player_num, embed_dim, padding_idx=PAD)


class ShotEmbedding(nn.Embedding):
    def __init__(self, shot_num, embed_dim):
        super(ShotEmbedding, self).__init__(shot_num, embed_dim, padding_idx=PAD)


class TransformerPredictor(nn.Module):
    def __init__(self, config):
        super(TransformerPredictor, self).__init__()
        self.transformer_decoder = TransformerDecoder(config)
        self.area_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['area_num'], bias=False)
        )
        self.shot_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['shot_num'], bias=False)
        )
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])


    def forward(self, input_shot, input_x, input_y, input_player, encoder_output, target_player, return_attns=False):
        if return_attns:
            x, decoder_self_attention_list, decoder_encoder_self_attention_list = self.transformer_decoder(input_shot, input_x, input_y, input_player, encoder_output, return_attns=return_attns)
        else:
            x = self.transformer_decoder(input_shot, input_x, input_y, input_player, encoder_output, return_attns=return_attns)
        embedded_player = self.player_embedding(target_player)
        x = (x + embedded_player)

        area_logits = self.area_decoder(x)
        shot_logits = self.shot_decoder(x)

        if return_attns:
            return area_logits, shot_logits, decoder_self_attention_list, decoder_encoder_self_attention_list
        else:
            return area_logits, shot_logits


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        self.n_layers = config['n_layers']
        self.hidden_size = config['shot_dim'] + config['area_dim']

        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1
        self.scale_emb = False
        self.d_model = d_model

        self.linear = nn.Linear(config['shot_dim']+config['area_dim'], config['player_dim'])

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round'])
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
            for _ in range(self.n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, input_shot, input_x, input_y, input_player, encoder_output, trg_mask=None, return_attns=False):
        decoder_self_attention_list, decoder_encoder_self_attention_list = [], []

        # triangular mask
        trg_mask = get_pad_mask(input_shot, PAD) & get_subsequent_mask(input_shot)

        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()
        
        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)

        embedded_area += embedded_player
        embedded_shot += embedded_player

        # add player embedding to shot and area embeddings
        embedded_shot_area = torch.cat((embedded_shot, embedded_area), dim=-1)       # (batch, seq_len, embed_dim)
        embedded_shot_area = self.linear(embedded_shot_area)

        # Forward
        if self.scale_emb:
            embedded_shot_area *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_embedding(embedded_shot_area, mode='decode'))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, encoder_output, slf_attn_mask=trg_mask)
            decoder_self_attention_list += [dec_slf_attn] if return_attns else []
            decoder_encoder_self_attention_list += [dec_enc_attn] if return_attns else []

        # (batch, seq_len, encode_dim), need to get last one to decode
        if return_attns:
            return dec_output, decoder_self_attention_list, decoder_encoder_self_attention_list
        return dec_output


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()

        self.n_layers = config['n_layers']
        self.hidden_size = config['shot_dim'] + config['area_dim']

        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1
        self.scale_emb = False
        self.d_model = d_model

        self.linear = nn.Linear(config['shot_dim']+config['area_dim'], config['player_dim'])

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round'])
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
            for _ in range(self.n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, input_shot, input_x, input_y, input_player, src_mask=None, return_attns=False):
        enc_slf_attn_list = []

        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()

        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)

        embedded_area += embedded_player
        embedded_shot += embedded_player

        # add player embedding to shot and area embeddings
        embedded_shot_area = torch.cat((embedded_shot, embedded_area), dim=-1)       # (batch, seq_len, embed_dim)

        embedded_shot_area = self.linear(embedded_shot_area)

        # Forward
        if self.scale_emb:
            embedded_shot_area *= self.d_model ** 0.5
        encode_output = self.dropout(self.position_embedding(embedded_shot_area, mode='encode'))
        encode_output = self.layer_norm(encode_output)

        for enc_layer in self.layer_stack:
            encode_output, enc_slf_attn = enc_layer(encode_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return encode_output, enc_slf_attn_list
        return encode_output