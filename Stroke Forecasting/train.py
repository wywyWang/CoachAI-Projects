from badmintoncleaner import prepare_dataset
import argparse

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn


def draw_loss(k_fold_index, record_total_loss, record_val_loss, config):
    x_steps = range(1, config['epochs']+1, 20)
    fig = plt.figure(figsize=(12, 6))
    plt.title("{} loss".format(config['model_type']))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim(0, 6)
    plt.xticks(x_steps)
    plt.grid()
    plt.plot(record_total_loss['total'], label='Train total loss')
    plt.plot(record_total_loss['shot'], label='Train shot CE loss')
    plt.plot(record_total_loss['area'], label='Train area NLL loss')

    if len(record_val_loss['total']) != 0:
        plt.plot(record_val_loss['total'], label='Val total loss')
        plt.plot(record_val_loss['entropy'], label='Val shot CE loss')
        plt.plot(record_val_loss['mse'], label='Val area MSE loss')
        plt.plot(record_val_loss['mae'], label='Val area MAE loss')

    plt.legend()
    plt.savefig(config['output_folder_name'] + str(k_fold_index) + '/' + 'loss.png')
    plt.close(fig)


def get_argument():
    opt = argparse.ArgumentParser()
    opt.add_argument("--model_type",
                        type=str,
                        choices=['LSTM', 'CFLSTM', 'Transformer', 'DMA_Nets', 'ShuttleNet', 'ours_rm_taa', 'ours_p2r', 'ours_r2p', 'DNRI'],
                        required=True,
                        help="model type")
    opt.add_argument("--output_folder_name",
                        type=str,
                        help="path to save model")
    opt.add_argument("--seed_value",
                        type=int,
                        default=42,
                        help="seed value")
    opt.add_argument("--max_ball_round",
                        type=int,
                        default=35,
                        help="max of ball round")
    opt.add_argument("--encode_length",
                        type=int,
                        default=4,
                        help="given encode length")
    opt.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="batch size")
    opt.add_argument("--lr",
                        type=int,
                        default=1e-4,
                        help="learning rate")
    opt.add_argument("--epochs",
                        type=int,
                        default=150,
                        help="epochs")
    opt.add_argument("--n_layers",
                        type=int,
                        default=1,
                        help="number of layers")
    opt.add_argument("--shot_dim",
                        type=int,
                        default=32,
                        help="dimension of shot")
    opt.add_argument("--area_num",
                        type=int,
                        default=5,
                        help="mux, muy, sx, sy, corr")
    opt.add_argument("--area_dim",
                        type=int,
                        default=32,
                        help="dimension of area")
    opt.add_argument("--player_dim",
                        type=int,
                        default=32,
                        help="dimension of player")
    opt.add_argument("--encode_dim",
                        type=int,
                        default=32,
                        help="dimension of hidden")
    opt.add_argument("--num_directions",
                        type=int,
                        default=1,
                        help="number of LSTM directions")
    opt.add_argument("--K",
                        type=int,
                        default=5,
                        help="Number of fold for dataset")
    opt.add_argument("--sample",
                        type=int,
                        default=10,
                        help="Number of samples for evaluation")
    config = vars(opt.parse_args())
    return config


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)    # gpu vars


config = get_argument()
model_type = config['model_type']

set_seed(config['seed_value'])

# Clean data and Prepare dataset
matches, total_train, total_val, total_test, config = prepare_dataset(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Model path: {}".format(config['output_folder_name']))

if not os.path.exists(config['output_folder_name']):
    os.makedirs(config['output_folder_name'])

k_fold_index = 0
for train_dataloader, test_dataloader in zip(total_train, total_test):
    k_fold_index += 1

    # create model
    if model_type == 'LSTM':
        from LSTM.GRU import GRUEncoder, GRUDecoder
        from LSTM.gru_runner import GRU_trainer
        encoder = GRUEncoder(config)
        decoder = GRUDecoder(config)
        encoder.area_embedding.weight = decoder.area_embedding.weight
        encoder.shot_embedding.weight = decoder.shot_embedding.weight
        encoder.player_embedding.weight = decoder.player_embedding.weight
        decoder.predictor.player_embedding.weight = decoder.player_embedding.weight
    elif model_type == 'CFLSTM':
        from CFLSTM.cf_lstm import CFLSTMEncoder, CFLSTMDecoder
        from CFLSTM.cf_lstm_runner import CFLSTM_trainer
        encoder = CFLSTMEncoder(config)
        decoder = CFLSTMDecoder(config)
        encoder.area_embedding.weight = decoder.area_embedding.weight
        encoder.shot_embedding.weight = decoder.shot_embedding.weight
        encoder.player_embedding.weight = decoder.player_embedding.weight
        decoder.predictor.player_embedding.weight = decoder.player_embedding.weight
    elif model_type == 'Transformer':
        from Transformer.transformer import TransformerEncoder, TransformerPredictor
        from Transformer.transformer_runner import transformer_trainer
        encoder = TransformerEncoder(config)
        decoder = TransformerPredictor(config)
        encoder.area_embedding.weight = decoder.transformer_decoder.area_embedding.weight
        encoder.shot_embedding.weight = decoder.transformer_decoder.shot_embedding.weight
        encoder.player_embedding.weight = decoder.transformer_decoder.player_embedding.weight
        decoder.player_embedding.weight = decoder.transformer_decoder.player_embedding.weight
    elif model_type == 'DNRI':
        from DNRI.DNRI import DNRIEncoder, DNRIDecoder
        from DNRI.dnri_runner import DNRI_trainer
        encoder = DNRIEncoder(config)
        decoder = DNRIDecoder(config)
        encoder.area_embedding.weight = decoder.area_embedding.weight
        encoder.shot_embedding.weight = decoder.shot_embedding.weight
        encoder.player_embedding.weight = decoder.player_embedding.weight
    elif model_type == 'DMA_Nets':
        from DMA_Nets.DMA_Nets import DMA_Nets_Encoder, DMA_Nets_Decoder
        from DMA_Nets.dma_nets_runner import DMA_Nets_trainer
        encoder = DMA_Nets_Encoder(config)
        decoder = DMA_Nets_Decoder(config)
        encoder.area_embedding.weight = decoder.area_embedding.weight
        encoder.shot_embedding.weight = decoder.shot_embedding.weight
        encoder.player_embedding.weight = decoder.player_embedding.weight
    elif model_type == 'ShuttleNet':
        from ShuttleNet.ShuttleNet import ShotGenEncoder, ShotGenPredictor
        from ShuttleNet.ShuttleNet_runner import shotGen_trainer
        encoder = ShotGenEncoder(config)
        decoder = ShotGenPredictor(config)
        encoder.area_embedding.weight = decoder.shotgen_decoder.area_embedding.weight
        encoder.shot_embedding.weight = decoder.shotgen_decoder.shot_embedding.weight
        encoder.player_embedding.weight = decoder.shotgen_decoder.player_embedding.weight
        decoder.player_embedding.weight = decoder.shotgen_decoder.player_embedding.weight
    elif model_type == 'ours_rm_taa':
        from ours_rm_taa.shotGen import ShotGenEncoder, ShotGenPredictor
        from ours_rm_taa.shotGen_runner import shotGen_trainer
        encoder = ShotGenEncoder(config)
        decoder = ShotGenPredictor(config)
        encoder.area_embedding.weight = decoder.shotgen_decoder.area_embedding.weight
        encoder.shot_embedding.weight = decoder.shotgen_decoder.shot_embedding.weight
        encoder.player_embedding.weight = decoder.shotgen_decoder.player_embedding.weight
        decoder.player_embedding.weight = decoder.shotgen_decoder.player_embedding.weight
    elif model_type == 'ours_p2r':
        from ours_p2r.shotGen_hie import ShotGenEncoder_hie, ShotGenPredictor_hie
        from ours_p2r.shotGen_runner_hie import shotGen_trainer_hie
        encoder = ShotGenEncoder_hie(config)
        decoder = ShotGenPredictor_hie(config)
        encoder.area_embedding.weight = decoder.shotgen_decoder.area_embedding.weight
        encoder.shot_embedding.weight = decoder.shotgen_decoder.shot_embedding.weight
        encoder.player_embedding.weight = decoder.shotgen_decoder.player_embedding.weight
        decoder.player_embedding.weight = decoder.shotgen_decoder.player_embedding.weight
    elif model_type == 'ours_r2p':
        from ours_r2p.shotGen_hie import ShotGenEncoder_hie, ShotGenPredictor_hie
        from ours_r2p.shotGen_runner_hie import shotGen_trainer_hie
        encoder = ShotGenEncoder_hie(config)
        decoder = ShotGenPredictor_hie(config)
        encoder.area_embedding.weight = decoder.shotgen_decoder.area_embedding.weight
        encoder.shot_embedding.weight = decoder.shotgen_decoder.shot_embedding.weight
        encoder.player_embedding.weight = decoder.shotgen_decoder.player_embedding.weight
        decoder.player_embedding.weight = decoder.shotgen_decoder.player_embedding.weight
    else:
        raise NotImplementedError

    # total model parameters
    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config['lr'])
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=config['lr'])

    encoder.to(device), decoder.to(device)

    criterion = {
        'entropy': nn.CrossEntropyLoss(ignore_index=0, reduction='sum'),
        'mse': nn.MSELoss(reduction='sum'),
        'mae': nn.L1Loss(reduction='sum')
    }
    for key, value in criterion.items():
        criterion[key].to(device)

    print("Model params: {}".format(total_params))

    # train model
    if model_type == 'LSTM':
        record_train_loss, record_val_loss = GRU_trainer(k_fold_index, train_dataloader, test_dataloader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, config=config, device=device)
    elif model_type == 'CFLSTM':
        record_train_loss, record_val_loss = CFLSTM_trainer(k_fold_index, train_dataloader, test_dataloader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, config=config, device=device)
    elif model_type == 'Transformer':
        record_train_loss, record_val_loss = transformer_trainer(k_fold_index, train_dataloader, test_dataloader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, config=config, device=device)
    elif model_type == 'DNRI':
        record_train_loss, record_val_loss = DNRI_trainer(k_fold_index, train_dataloader, test_dataloader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, config=config, device=device)
    elif model_type == 'DMA_Nets':
        record_train_loss, record_val_loss = DMA_Nets_trainer(k_fold_index, train_dataloader, test_dataloader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, config=config, device=device)
    elif model_type == 'ShuttleNet':
        record_train_loss, record_val_loss = shotGen_trainer(k_fold_index, train_dataloader, test_dataloader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, config=config, device=device)
    elif model_type == 'ours_rm_taa':
        record_train_loss, record_val_loss = shotGen_trainer(k_fold_index, train_dataloader, test_dataloader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, config=config, device=device)
    elif model_type == 'ours_p2r':
        record_train_loss, record_val_loss = shotGen_trainer_hie(k_fold_index, train_dataloader, test_dataloader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, config=config, device=device)
    elif model_type == 'ours_r2p':
        record_train_loss, record_val_loss = shotGen_trainer_hie(k_fold_index, train_dataloader, test_dataloader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, config=config, device=device)
    else:
        raise NotImplementedError

    # draw loss
    draw_loss(k_fold_index, record_train_loss, record_val_loss, config)
