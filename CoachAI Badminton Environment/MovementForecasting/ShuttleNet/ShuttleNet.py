import torch
import torch.nn as nn
import torch.nn.functional as F
from ShuttleNet.ShuttleNet_layers import EncoderLayer, DecoderLayer, GatedFusionLayer
from ShuttleNet.ShuttleNet_embedding import PositionalEncoding


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


class ShotGenDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.area_embedding = nn.Linear(2, config['area_dim'])
        # self.shot_embedding = ShotEmbedding(config['type_num'], config['type_dim'])
        # self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])

        # self.player_embedding = nn.Embedding(config['player_num'] * 4 + 1, config['player_dim'])
        self.player_embedding = nn.Embedding(config['player_num'], config['player_dim'])

        self.type_embedding = nn.Embedding(config['type_num'], config['type_dim'])
        self.self_coordination_transform = nn.Linear(2, config['location_dim'])
        self.coordination_transform = nn.Linear(config['location_dim']*2, config['location_dim'])
        self.player_num = config['player_num']

        n_heads = 2
        d_k = config['hidden_size']
        d_v = config['hidden_size']
        d_model = config['hidden_size']
        d_inner = config['hidden_size'] * 2
        dropout = 0.1
        self.d_model = d_model

        self.position_embedding = PositionalEncoding(config['type_dim'], config['encode_length'], n_position=config['max_length'])
        self.dropout = nn.Dropout(p=dropout)

        self.global_layer = DecoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = DecoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

        self.gated_fusion = GatedFusionLayer(d_model, d_model, config['encode_length'], config['max_length'])

        self.location_player = nn.Linear(config['location_dim']+config['player_dim'], config['hidden_size'])
        self.type_player = nn.Linear(config['type_dim']+config['player_dim'], config['hidden_size'])

    def forward(self, player, player_A_x, player_A_y, player_B_x, player_B_y, shot_type, encode_local_output, encode_global_A, encode_global_B, trg_mask=None, return_attns=False, two_player=None):
        decoder_self_attention_list, decoder_encoder_self_attention_list = [], []

        player_embedding = self.player_embedding(player)
        two_player = self.player_embedding(two_player)

        player_A_coordination = torch.cat((player_A_x.unsqueeze(2), player_A_y.unsqueeze(2)), dim=2).float()
        player_B_coordination = torch.cat((player_B_x.unsqueeze(2), player_B_y.unsqueeze(2)), dim=2).float()
        
        player_A_coordination = self.self_coordination_transform(player_A_coordination)
        player_B_coordination = self.self_coordination_transform(player_B_coordination)
        player_A_coordination = F.relu(player_A_coordination)
        player_B_coordination = F.relu(player_B_coordination)

        player_A_coordination = torch.cat((player_A_coordination, two_player[:, 0:1, :].repeat(1, player.size(1), 1)), dim=-1)
        player_B_coordination = torch.cat((player_B_coordination, two_player[:, 1:2, :].repeat(1, player.size(1), 1)), dim=-1)

        player_A_coordination = self.location_player(player_A_coordination)
        player_B_coordination = self.location_player(player_B_coordination)

        coordination_sequence = torch.cat((player_A_coordination, player_B_coordination), dim=2)
        coordination_transform = self.coordination_transform(coordination_sequence)   

        # split player only for masking
        mask_A = shot_type[:, ::2]
        mask_B = shot_type[:, 1::2]

        # triangular mask
        trg_local_mask = get_pad_mask(shot_type) & get_subsequent_mask(shot_type)
        trg_global_A_mask = get_pad_mask(mask_A) & get_subsequent_mask(mask_A)
        trg_global_B_mask = get_pad_mask(mask_B) & get_subsequent_mask(mask_B)

        # interleave the player and opponent location
        type_embedding = self.type_embedding(shot_type)
        type_embedding = torch.cat((type_embedding, player_embedding), dim=-1)
        type_embedding = self.type_player(type_embedding)

        # split player
        h_a_A = coordination_transform[:, ::2]
        h_a_B = coordination_transform[:, 1::2]
        h_s_A = type_embedding[:, ::2]
        h_s_B = type_embedding[:, 1::2]

        # local
        decode_output_area = self.dropout(self.position_embedding(coordination_transform, mode='decode'))
        decode_output_shot = self.dropout(self.position_embedding(type_embedding, mode='decode'))
        # global
        decode_output_area_A = self.dropout(self.position_embedding(h_a_A, mode='decode'))
        decode_output_area_B = self.dropout(self.position_embedding(h_a_B, mode='decode'))
        decode_output_shot_A = self.dropout(self.position_embedding(h_s_A, mode='decode'))
        decode_output_shot_B = self.dropout(self.position_embedding(h_s_B, mode='decode'))

        decode_global_A, dec_slf_attn_A, dec_enc_attn_A, disentangled_weight_A = self.global_layer(decode_output_area_A, decode_output_shot_A, encode_global_A, slf_attn_mask=trg_global_A_mask, return_attns=return_attns)
        if decode_output_area_B.shape[1] != 0:
            decode_global_B, dec_slf_attn_B, dec_enc_attn_B, disentangled_weight_B = self.global_layer(decode_output_area_B, decode_output_shot_B, encode_global_B, slf_attn_mask=trg_global_B_mask, return_attns=return_attns)

        decode_local_output, dec_slf_attn, dec_enc_attn, disentangled_weight_local = self.local_layer(decode_output_area, decode_output_shot, encode_local_output, slf_attn_mask=trg_local_mask, return_attns=return_attns)
        decoder_self_attention_list = dec_slf_attn if return_attns else []
        decoder_encoder_self_attention_list = dec_enc_attn if return_attns else []

        if decode_output_area_B.shape[1] != 0:
            decode_output_A = alternatemerge(decode_global_A, decode_global_A, decode_local_output.shape[1], 'A')
            decode_output_B = alternatemerge(decode_global_B, decode_global_B, decode_local_output.shape[1], 'B')
        else:
            decode_output_A = decode_global_A.clone()
            decode_output_B = torch.zeros(decode_local_output.shape, device=decode_local_output.device)
        decode_output = self.gated_fusion(decode_output_A, decode_output_B, decode_local_output)

        # (batch, seq_len, embed_dim)
        if return_attns:
            return decode_output, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        return decode_output


class ShotGenPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shotgen_decoder = ShotGenDecoder(config)
        self.area_decoder = nn.Sequential(
            nn.Linear(config['hidden_size'], 10, bias=False)
        )
        self.shot_decoder = nn.Sequential(
            nn.Linear(config['hidden_size'], config['type_num'], bias=False),
        )
        # self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])

    def forward(self, player, player_A_x, player_A_y, player_B_x, player_B_y, shot_type, encode_local_output, encode_global_A, encode_global_B, return_attns=False, two_player=None):
        # embedded_target_player = self.player_embedding(target_player)
        if return_attns:
            decode_output, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local = self.shotgen_decoder(player, player_A_x, player_A_y, player_B_x, player_B_y, shot_type, encode_local_output, encode_global_A, encode_global_B, return_attns=return_attns)
        else:
            decode_output = self.shotgen_decoder(player, player_A_x, player_A_y, player_B_x, player_B_y, shot_type, encode_local_output, encode_global_A, encode_global_B, return_attns, two_player=two_player)

        # decode_output = (decode_output + embedded_target_player)

        area_logits = self.area_decoder(decode_output)
        shot_logits = self.shot_decoder(decode_output)

        if return_attns:
            return area_logits, shot_logits, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        else:
            return area_logits, shot_logits


class ShotGenEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.area_embedding = nn.Linear(2, config['area_dim'])
        # self.shot_embedding = ShotEmbedding(config['type_num'], config['type_dim'])
        # self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])

        # self.player_embedding = nn.Embedding(config['player_num'] * 4 + 1, config['player_dim'])
        self.player_embedding = nn.Embedding(config['player_num'], config['player_dim'])
        self.type_embedding = nn.Embedding(config['type_num'], config['type_dim'])
        self.self_coordination_transform = nn.Linear(2, config['location_dim'])
        self.coordination_transform = nn.Linear(config['location_dim']*2, config['location_dim'])
        self.player_num = config['player_num']
        self.encode_length = config['encode_length']
        n_heads = 2
        d_k = config['hidden_size']
        d_v = config['hidden_size']
        d_model = config['hidden_size']
        d_inner = config['hidden_size'] * 2
        dropout = 0.1
        self.d_model = d_model

        self.position_embedding = PositionalEncoding(config['type_dim'], config['encode_length'], n_position=config['max_length'])
        self.dropout = nn.Dropout(p=dropout)

        self.global_layer = EncoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = EncoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

        self.location_player = nn.Linear(config['location_dim']+config['player_dim'], config['hidden_size'])
        self.type_player = nn.Linear(config['type_dim']+config['player_dim'], config['hidden_size'])

    def forward(self, player, player_A_x, player_A_y, player_B_x, player_B_y, shot_type, src_mask=None, return_attns=False, two_player=None):
        enc_slf_attn_list = []

        player_embedding = self.player_embedding(player)
        two_player = self.player_embedding(two_player)
        
        player_A_coordination = torch.cat((player_A_x.unsqueeze(2), player_A_y.unsqueeze(2)), dim=2).float()
        player_B_coordination = torch.cat((player_B_x.unsqueeze(2), player_B_y.unsqueeze(2)), dim=2).float()
        player_A_coordination = self.self_coordination_transform(player_A_coordination)
        player_B_coordination = self.self_coordination_transform(player_B_coordination)
        player_A_coordination = F.relu(player_A_coordination)
        player_B_coordination = F.relu(player_B_coordination)

        player_A_coordination = torch.cat((player_A_coordination, two_player[:, 0:1, :].repeat(1, player.size(1), 1)), dim=-1)
        player_B_coordination = torch.cat((player_B_coordination, two_player[:, 1:2, :].repeat(1, player.size(1), 1)), dim=-1)

        player_A_coordination = self.location_player(player_A_coordination)
        player_B_coordination = self.location_player(player_B_coordination)

        coordination_sequence = torch.cat((player_A_coordination, player_B_coordination), dim=2)
        coordination_transform = self.coordination_transform(coordination_sequence)

        type_embedding = self.type_embedding(shot_type)
        type_embedding = torch.cat((type_embedding, player_embedding), dim=-1)
        type_embedding = self.type_player(type_embedding)
        
        # split player
        h_a_A = coordination_transform[:, ::2]
        h_a_B = coordination_transform[:, 1::2]
        h_s_A = type_embedding[:, ::2]
        h_s_B = type_embedding[:, 1::2]

        # local
        encode_output_area = self.dropout(self.position_embedding(coordination_transform, mode='encode'))
        encode_output_shot = self.dropout(self.position_embedding(type_embedding, mode='encode'))
        # global
        encode_output_area_A = self.dropout(self.position_embedding(h_a_A, mode='encode'))
        encode_output_area_B = self.dropout(self.position_embedding(h_a_B, mode='encode'))
        encode_output_shot_A = self.dropout(self.position_embedding(h_s_A, mode='encode'))
        encode_output_shot_B = self.dropout(self.position_embedding(h_s_B, mode='encode'))

        encode_global_A, enc_slf_attn_A = self.global_layer(encode_output_area_A, encode_output_shot_A, slf_attn_mask=src_mask)
        encode_global_B, enc_slf_attn_B = self.global_layer(encode_output_area_B, encode_output_shot_B, slf_attn_mask=src_mask)
        
        encode_local_output, enc_slf_attn = self.local_layer(encode_output_area, encode_output_shot, slf_attn_mask=src_mask)

        if return_attns:
            return encode_local_output, encode_global_A, encode_global_B, enc_slf_attn_list
        return encode_local_output, encode_global_A, encode_global_B