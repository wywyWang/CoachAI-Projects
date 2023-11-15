import torch.nn as nn
import torch
import torch.nn.functional as F
#from ShuttleNet.ShuttleNet_submodules import TypeAreaMultiHeadAttention#,MultiHeadAttention
from ShuttleNet.PositionwiseFeedForward import PositionwiseFeedForward
from ShuttleNet.MultiHeadAttention import MultiHeadAttention, TypeAreaMultiHeadAttention
from ShuttleNet.PositionalEncoding import PositionalEncoding
from ShuttleNet.Embedding import EmbeddingLayer
from ShuttleNet.GatedFusion import GatedFusionLayer

PAD = 0

def get_pad_mask(seq):
    return (seq != PAD).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


def alternatemerge(seq_A, seq_B, merge_len, player):
    # (batch, seq_len, dim)
    seq_len = seq_A.shape[1]
    merged_seq = torch.zeros(seq_A.shape[0], merge_len, seq_A.shape[2])

    if seq_len * 2 == (merge_len - 1):
        # if seq_len is odd and B will shorter, e.g., merge = 5, A = 3, B = 2
        merged_seq[:, 1::2, :] = seq_A[:, :seq_len, :]
        merged_seq[:, 2::2, :] = seq_B[:, :seq_len, :]
    elif (seq_len * 2 - 1) == merge_len:
        # if seq_len is odd and A will longer, e.g., merge = 5, A = 3, B = 2
        merged_seq[:, ::2, :] = seq_A[:, :seq_len, :]
        merged_seq[:, 1::2, :] = seq_B[:, :merge_len-seq_len, :]
    elif seq_len * 2 == merge_len:
        if player == 'A':
            merged_seq[:, ::2, :] = seq_A[:, :seq_len, :]
            merged_seq[:, 1::2, :] = seq_B[:, :seq_len, :]
        elif player == 'B':
            merged_seq[:, 1::2, :] = seq_A[:, :seq_len, :]
            merged_seq[:, 2::2, :] = seq_B[:, :seq_len-1, :]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return merged_seq.cuda(seq_A.device)

class DecoderModule(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, config, dropout=0.1):
        super().__init__()

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round'])
        self.dropout = nn.Dropout(p=dropout)

        self.multiHeadAttention_decoder = TypeAreaMultiHeadAttention(n_head, d_model, dropout=dropout)
        self.multiHeadAttention_endecoder = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.FFN = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, h_a, h_s, encode_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        decode_area = self.dropout(self.position_embedding(h_a, mode='decode'))
        decode_shot = self.dropout(self.position_embedding(h_s, mode='decode'))

        # Multi-head type-area attention (decoder)
        decode_output = self.multiHeadAttention_decoder(decode_area, decode_area, decode_area, decode_shot, decode_shot, decode_shot, mask=slf_attn_mask)
        # Multi-head attention (encoder)
        decode_output = self.multiHeadAttention_endecoder(decode_output, encode_output, encode_output, mask=dec_enc_attn_mask)
        #FFN
        decode_output = self.FFN(decode_output)
        return decode_output

class PredictorLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.area_decoder = nn.Sequential(nn.Linear(config['encode_dim'], config['area_num'], bias=False))
        self.shot_decoder = nn.Sequential(nn.Linear(config['encode_dim'], config['shot_num'], bias=False))
        self.player_embedding = nn.Embedding(config['player_num'], config['player_dim'], padding_idx=PAD)

    def forward(self, decode_output, target_player):
        embedded_target_player = self.player_embedding(target_player)

        decode_output = (decode_output + embedded_target_player)

        area_logits = self.area_decoder(decode_output)
        shot_logits = self.shot_decoder(decode_output)

        return area_logits, shot_logits

class ShotGenDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = EmbeddingLayer(config['area_dim'], config['shot_dim'], config['player_dim'], config['shot_num'], config['player_num'])

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round'])
        self.dropout = nn.Dropout(p=dropout)

        self.playerDecode = DecoderModule(d_model, d_inner, n_heads, d_k, d_v, config, dropout=dropout)
        self.rallyDecode  = DecoderModule(d_model, d_inner, n_heads, d_k, d_v, config, dropout=dropout)

        self.gatedFusion = GatedFusionLayer(d_model, config['encode_length'], config['max_ball_round'])

        self.predictor = PredictorLayer(config)

    def forward(self, shot, x, y, player, encoderOutput_rally, encoderOutput_A, encoderOutput_B, target_player, trg_mask=None):
        area = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), dim=-1).float()

        # split player only for masking
        mask_A = shot[:, ::2]  # even pos
        mask_B = shot[:, 1::2]  # odd pos

        # triangular mask
        trg_local_mask = get_pad_mask(shot) & get_subsequent_mask(shot)
        trg_global_A_mask = get_pad_mask(mask_A) & get_subsequent_mask(mask_A)
        trg_global_B_mask = get_pad_mask(mask_B) & get_subsequent_mask(mask_B)

        embedded_area, embedded_shot, embedded_player = self.embedding(area, shot, player)

        h_a = embedded_area + embedded_player
        h_s = embedded_shot + embedded_player

        # split player
        h_a_A = h_a[:, ::2]
        h_a_B = h_a[:, 1::2]
        h_s_A = h_s[:, ::2]
        h_s_B = h_s[:, 1::2]

        decode_A = self.playerDecode(h_a_A, h_s_A, encoderOutput_A, slf_attn_mask=trg_global_A_mask)
        if h_a_B.shape[1] != 0:
            decode_B = self.playerDecode(h_a_B, h_s_B, encoderOutput_B, slf_attn_mask=trg_global_B_mask)

        decode_rally = self.rallyDecode(h_a, h_s, encoderOutput_rally, slf_attn_mask=trg_local_mask)

        if h_a_B.shape[1] != 0:
            decode_A = alternatemerge(decode_A, decode_A, decode_rally.shape[1], 'A')
            decode_B = alternatemerge(decode_B, decode_B, decode_rally.shape[1], 'B')
        else:
            decode_A = decode_A.clone()
            decode_B = torch.zeros(decode_rally.shape, device=decode_rally.device)
        decode_output = self.gatedFusion(decode_A, decode_B, decode_rally)

        area_logits, shot_logits = self.predictor(decode_output, target_player)

        return area_logits, shot_logits

