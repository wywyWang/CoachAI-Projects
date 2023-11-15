import torch.nn as nn
import torch
import torch.nn.functional as F
from ShuttleNet.ShuttleNet_submodules import TypeAreaMultiHeadAttention, MultiHeadAttention, PositionwiseFeedForward
# from ShuttleNet.ShuttleNet_embedding import PositionalEncoding
from ShuttleNet.PositionalEncoding import PositionalEncoding
from ShuttleNet.Embedding import EmbeddingLayer

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

class GatedFusionLayer(nn.Module):
    def __init__(self, d, d_inner, encode_length, n_position=200):
        super().__init__()
        n = 3
        self.hidden1 = nn.Linear(d, d_inner, bias=False)
        self.hidden2 = nn.Linear(d, d_inner, bias=False)
        self.hidden3 = nn.Linear(d, d_inner, bias=False)
        self.gated1 = nn.Linear(d_inner*n, d, bias=False)
        self.gated2 = nn.Linear(d_inner*n, d, bias=False)
        self.gated3 = nn.Linear(d_inner*n, d, bias=False)

        self.decode_length = n_position - encode_length

        self.w_A = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def project(self, x_A, x_B, x_L):
        h_A = self.tanh(self.hidden1(x_A))
        h_B = self.tanh(self.hidden2(x_B))
        h_L = self.tanh(self.hidden3(x_L))

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
        alpha_A = self.sigmoid(self.gated1(x))
        alpha_B = self.sigmoid(self.gated2(x))
        alpha_L = self.sigmoid(self.gated3(x))

        h_A, h_B, h_L = self.project(x_A, x_B, x_L)

        z1 = beta_A * alpha_A * h_A
        z2 = beta_B * alpha_B * h_B
        z3 = beta_L * alpha_L * h_L

        return self.sigmoid(z1 + z2 + z3)

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, config, dropout=0.1):
        super().__init__()
        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round'])
        self.dropout = nn.Dropout(p=dropout)

        self.multiHeadAttention_decoder = TypeAreaMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.multiHeadAttention_endecoder = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
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
        #self.area_embedding = nn.Linear(2, config['area_dim'])
        #self.shot_embedding = nn.Embedding(config['shot_num'], config['shot_dim'], padding_idx=PAD)
        #self.player_embedding = nn.Embedding(config['player_num'], config['player_dim'], padding_idx=PAD)

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round'])
        self.dropout = nn.Dropout(p=dropout)

        self.player_layer = DecoderLayer(d_model, d_inner, n_heads, d_k, d_v, config, dropout=dropout)
        self.rally_layer = DecoderLayer(d_model, d_inner,n_heads, d_k, d_v, config, dropout=dropout)

        self.gatedFusion = GatedFusionLayer(d_model, d_model, config['encode_length'], config['max_ball_round'])

        self.predictor = PredictorLayer(config)

    def forward(self, input_shot, input_x, input_y, input_player, encode_local_output, encode_global_A, encode_global_B, target_player, trg_mask=None):
        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()

        # split player only for masking
        mask_A = input_shot[:, ::2]  # even pos
        mask_B = input_shot[:, 1::2]  # odd pos

        # triangular mask
        trg_local_mask = get_pad_mask(input_shot) & get_subsequent_mask(input_shot)
        trg_global_A_mask = get_pad_mask(mask_A) & get_subsequent_mask(mask_A)
        trg_global_B_mask = get_pad_mask(mask_B) & get_subsequent_mask(mask_B)

        #embedded_area = F.relu(self.area_embedding(area))
        #embedded_shot = self.shot_embedding(input_shot)
        #embedded_player = self.player_embedding(input_player)
        embedded_area, embedded_shot, embedded_player = self.embedding(area, input_shot, input_player)

        h_a = embedded_area + embedded_player
        h_s = embedded_shot + embedded_player

        # split player
        h_a_A = h_a[:, ::2]
        h_a_B = h_a[:, 1::2]
        h_s_A = h_s[:, ::2]
        h_s_B = h_s[:, 1::2]

        decode_A = self.player_layer(h_a_A, h_s_A, encode_global_A, slf_attn_mask=trg_global_A_mask)
        if h_a_B.shape[1] != 0:
            decode_B = self.player_layer(h_a_B, h_s_B, encode_global_B, slf_attn_mask=trg_global_B_mask)

        decode_rally = self.rally_layer(h_a, h_s, encode_local_output, slf_attn_mask=trg_local_mask)

        if h_a_B.shape[1] != 0:
            decode_A = alternatemerge(decode_A, decode_A, decode_rally.shape[1], 'A')
            decode_B = alternatemerge(decode_B, decode_B, decode_rally.shape[1], 'B')
        else:
            decode_A = decode_A.clone()
            decode_B = torch.zeros(decode_rally.shape, device=decode_rally.device)
        decode_output = self.gatedFusion(decode_A, decode_B, decode_rally)

        area_logits, shot_logits = self.predictor(decode_output, target_player)

        return area_logits, shot_logits

