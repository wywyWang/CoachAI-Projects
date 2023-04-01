from badmintoncleaner import prepare_dataset
from utils import draw_loss
import argparse
import os
import torch
import torch.nn as nn


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
                        default=70,
                        help="max of ball round (hard code in this sample code)")
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
    opt.add_argument("--gpu_num",
                        type=int,
                        default=0,
                        help="Selected GPU number")
    config = vars(opt.parse_args())
    return config


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)    # gpu vars


if __name__ == "__main__":
    config = get_argument()
    config['data_folder'] = '../data/'
    config['model_folder'] = './model/'
    model_type = config['model_type']
    set_seed(config['seed_value'])

    # Clean data and Prepare dataset
    config, train_dataloader, val_dataloader, test_dataloader, train_matches, val_matches, test_matches = prepare_dataset(config)

    device = torch.device(f"cuda:{config['gpu_num']}" if torch.cuda.is_available() else "cpu")
    print("Model path: {}".format(config['output_folder_name']))
    if not os.path.exists(config['output_folder_name']):
        os.makedirs(config['output_folder_name'])

    # read model
    from ShuttleNet.ShuttleNet import ShotGenEncoder, ShotGenPredictor
    from ShuttleNet.ShuttleNet_runner import shotGen_trainer
    encoder = ShotGenEncoder(config)
    decoder = ShotGenPredictor(config)
    encoder.area_embedding.weight = decoder.shotgen_decoder.area_embedding.weight
    encoder.shot_embedding.weight = decoder.shotgen_decoder.shot_embedding.weight
    encoder.player_embedding.weight = decoder.shotgen_decoder.player_embedding.weight
    decoder.player_embedding.weight = decoder.shotgen_decoder.player_embedding.weight

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config['lr'])
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=config['lr'])
    encoder.to(device), decoder.to(device)

    criterion = {
        'entropy': nn.CrossEntropyLoss(ignore_index=0, reduction='sum'),
        'mae': nn.L1Loss(reduction='sum')
    }
    for key, value in criterion.items():
        criterion[key].to(device)

    record_train_loss = shotGen_trainer(data_loader=train_dataloader, encoder=encoder, decoder=decoder, criterion=criterion, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer, config=config, device=device)

    draw_loss(record_train_loss, config)