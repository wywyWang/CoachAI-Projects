import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
from badmintoncleaner import prepare_dataset
from ShuttleNet.ShuttleNet_runner import shotGen_trainer
import argparse

import matplotlib
matplotlib.use('AGG')


def draw_loss(k_fold_index, record_total_loss, record_val_loss, config):
    x_steps = range(1, config['epochs']+1, 20)
    fig = plt.figure(figsize=(12, 6))
    # plt.title("{} loss".format(config['model_type']))
    plt.title("ShuttleNet loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim(0, 15)
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
    plt.savefig(f"{config['output_folder_name']}{k_fold_index}/loss.png")
    plt.close(fig)

GMM_NUM = 5
def get_argument():
    opt = argparse.ArgumentParser()
    opt.add_argument("--output_folder_name",
                     type=str,
                     default='continuousGMM_cannotReach',
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
                     default=2,
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
                     default=350,
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
    opt.add_argument("--num_GMM",
                     type=int,
                     default=7,
                     help="number of GMM")
    config = vars(opt.parse_args())
    config['area_num'] *= config['num_GMM']
    config['output_folder_name'] = f'{config["output_folder_name"]}_{config["num_GMM"]}GMM{config["epochs"]}'
    return config


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)    # gpu vars


def main():
    config = get_argument()

    set_seed(config['seed_value'])

    # Clean data and Prepare dataset
    matches, total_train, total_val, total_test, config = prepare_dataset(config)

    #test device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # output path
    print(f"Model path: {config['output_folder_name']}")

    if not os.path.exists(config['output_folder_name']):
        os.makedirs(config['output_folder_name'])

    k_fold_index = 0
    for train_dataloader, test_dataloader in zip(total_train, total_test):
        k_fold_index += 1

        print(k_fold_index)
        # train model
        record_train_loss, record_val_loss = shotGen_trainer(k_fold_index, train_dataloader, test_dataloader, config, device=device)

        # draw loss
        draw_loss(k_fold_index, record_train_loss, record_val_loss, config)


if __name__ == '__main__':
    main()
