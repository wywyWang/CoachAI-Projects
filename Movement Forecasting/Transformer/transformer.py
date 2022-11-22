import torch
import torch.nn as nn
import torch.nn.functional as F
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
            nn.Linear(config['hidden_size'], 10, bias=False)
        )
        self.shot_decoder = nn.Sequential(
            nn.Linear(config['hidden_size'], config['type_num'], bias=False)
        )
        # self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])


    def forward(self, player, player_A_x, player_A_y, player_B_x, player_B_y, shot_type, encoder_output, return_attns=False, two_player=None):
        if return_attns:
            x, decoder_self_attention_list, decoder_encoder_self_attention_list = self.transformer_decoder(input_shot, input_x, input_y, input_player, encoder_output, return_attns=return_attns)
        else:
            decode_output = self.transformer_decoder(player, player_A_x, player_A_y, player_B_x, player_B_y, shot_type, encoder_output, return_attns=return_attns, two_player=two_player)
        # embedded_player = self.player_embedding(target_player)
        # x = (x + embedded_player)

        area_logits = self.area_decoder(decode_output)
        shot_logits = self.shot_decoder(decode_output)

        if return_attns:
            return area_logits, shot_logits, decoder_self_attention_list, decoder_encoder_self_attention_list
        else:
            return area_logits, shot_logits


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        # self.area_embedding = nn.Linear(2, config['area_dim'])
        # self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        # self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])

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

        self.location_player = nn.Linear(config['location_dim']+config['player_dim'], config['hidden_size'])
        self.type_player = nn.Linear(config['type_dim']+config['player_dim'], config['hidden_size'])

        self.position_embedding = PositionalEncoding(config['type_dim'], config['encode_length'], n_position=config['max_length'])
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
            for _ in range(1)])
        self.model_input_linear = nn.Linear(config['location_dim'] + config['type_dim'], config['hidden_size'])
    def forward(self, player, player_A_x, player_A_y, player_B_x, player_B_y, shot_type, encoder_output, trg_mask=None, return_attns=False, two_player=None):
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
        # triangular mask
        trg_mask = get_pad_mask(shot_type, PAD) & get_subsequent_mask(shot_type)

        coordination_sequence = torch.cat((player_A_coordination, player_B_coordination), dim=2)
        coordination_transform = self.coordination_transform(coordination_sequence)

        type_embedding = self.type_embedding(shot_type)
        type_embedding = torch.cat((type_embedding, player_embedding), dim=-1)
        type_embedding = self.type_player(type_embedding)

        type_embedding = self.dropout(self.position_embedding(type_embedding, mode='decode'))
        coordination_transform = self.dropout(self.position_embedding(coordination_transform, mode='decode'))

        model_input = torch.cat((coordination_transform, type_embedding), dim=-1)    
        dec_output = self.model_input_linear(model_input)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, encoder_output, slf_attn_mask=trg_mask)
            decoder_self_attention_list += [dec_slf_attn] if return_attns else []
            decoder_encoder_self_attention_list += [dec_enc_attn] if return_attns else []


        if return_attns:
            return dec_output, decoder_self_attention_list, decoder_encoder_self_attention_list
        return dec_output


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()

        # self.area_embedding = nn.Linear(2, config['area_dim'])
        # self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        # self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])

        self.player_embedding = nn.Embedding(config['player_num'], config['player_dim'])
        self.type_embedding = nn.Embedding(config['type_num'], config['type_dim'])
        self.self_coordination_transform = nn.Linear(2, config['location_dim'])
        self.location_player = nn.Linear(config['player_dim']+config['location_dim'], config['hidden_size'])
        self.type_player = nn.Linear(config['player_dim']+config['type_dim'], config['hidden_size'])
        self.coordination_transform = nn.Linear(config['location_dim']*2, config['location_dim'])
        self.player_num = config['player_num']
        self.encode_length = config['encode_length']

        self.location_player = nn.Linear(config['location_dim']+config['player_dim'], config['hidden_size'])
        self.type_player = nn.Linear(config['type_dim']+config['player_dim'], config['hidden_size'])

        n_heads = 2
        d_k = config['hidden_size']
        d_v = config['hidden_size']
        d_model = config['hidden_size']
        d_inner = config['hidden_size'] * 2
        dropout = 0.1
        self.d_model = d_model

        self.position_embedding = PositionalEncoding(config['type_dim'], config['encode_length'], n_position=config['max_length'])
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
            for _ in range(1)])
        self.model_input_linear = nn.Linear(config['location_dim'] + config['type_dim'], config['hidden_size'])
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

        type_embedding = self.dropout(self.position_embedding(type_embedding, mode='encode'))
        coordination_transform = self.dropout(self.position_embedding(coordination_transform, mode='encode'))

        model_input = torch.cat((coordination_transform, type_embedding), dim=-1)
        encode_output = self.model_input_linear(model_input)

        for enc_layer in self.layer_stack:
            encode_output, enc_slf_attn = enc_layer(encode_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return encode_output, enc_slf_attn_list
        return encode_output