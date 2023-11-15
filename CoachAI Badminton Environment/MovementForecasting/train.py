import argparse
import torch
import numpy as np
import random
import torch.nn as nn
from datetime import datetime
import os

from prepare_dataset import prepare_dataset
from utils import save_args_file


def main():
    args = argparse.ArgumentParser()

    # prepare data
    args.add_argument("--input_data_folder_path", type=str, default="./data/")
    args.add_argument("--match_list_csv", type=str, default="match.csv")
    args.add_argument("--homography_matrix_list_csv", type=str, default="homography.csv")
    args.add_argument("--prepared_data_output_path", type=str, default="./data/dataset.csv")
    args.add_argument("--already_have_data", type=int, default=1)
    args.add_argument("--preprocessed_data_path", type=str, default="./data/dataset.csv")
    args.add_argument("--train_ratio", type=float, default=0.8)
    args.add_argument("--valid_ratio", type=float, default=0)
    args.add_argument("--max_length", type=int, default=35)

    # training
    args.add_argument("--seed", type=int, default=22)
    args.add_argument("--train_batch_size", type=int, default=32)
    args.add_argument("--valid_batch_size", type=int, default=1)
    args.add_argument("--test_batch_size", type=int, default=1)
    args.add_argument("--hidden_size", type=int, default=16)
    args.add_argument("--model_type", type=str, default='DyMF')
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--player_dim", type=int, default=16)
    args.add_argument("--type_dim", type=int, default=16)
    args.add_argument("--location_dim", type=int, default=16)
    args.add_argument("--num_layer", type=int, default=2)

    args.add_argument("--epochs", type=int, default=100)
    args.add_argument("--encode_length", type=int, default=2)
    args.add_argument("--dropout", type=float, default=0.1)

    args.add_argument("--num_basis", type=int, default=3)
    args.add_argument("--num_GMM", type=int, default=5)

    # ablation
    args.add_argument("--use_complete_graph", type=int, default=0)
    args.add_argument("--without_dynamic_gcn", type=int, default=0)
    args.add_argument("--without_tactical_fusion", type=int, default=0)
    args.add_argument("--without_player_style_fusion", type=int, default=0)
    args.add_argument("--without_rally_fusion", type=int, default=0)
    args.add_argument("--without_style_fusion", type=int, default=0)
    args.add_argument("--without_refer", type=int, default=0)

    # save model
    args.add_argument("--output_model_path", type=str, default='./model/')
    args.add_argument("--model_folder", type=str, default=None)

    # sample
    args.add_argument("--sample_num", type=int, default=1)

    args = args.parse_args()
    args = vars(args)

    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args['model_folder'] == None:
        args['model_folder'] = './model/' +  args['model_type'] + '_' + str(args['encode_length']) + '_' + str(datetime.now().strftime("%Y%m%d"))

    train_dataloader, valid_dataloader, test_dataloader, args = prepare_dataset(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args['model_type'] == 'DNRI':
        from DNRI.model import Encoder, Decoder
        from DNRI.runner import train
        encoder = Encoder(args)
        decoder = Decoder(args)
        encoder.player_embedding.weight = decoder.player_embedding.weight
        encoder.coordination_transform.weight = decoder.coordination_transform.weight 

    if args['model_type'] == 'LSTM':
        from LSTM.model import Encoder, Decoder
        from LSTM.runner import train
        encoder = Encoder(args)
        decoder = Decoder(args)
        encoder.player_embedding.weight = decoder.player_embedding.weight
        encoder.type_embedding.weight = decoder.type_embedding.weight
        encoder.coordination_transform.weight = decoder.coordination_transform.weight

    if args['model_type'] == 'DyMF':
        if args['use_complete_graph'] == 1:
            from DyMF.model_complete import Encoder, Decoder
            from DyMF.runner import train
        elif args['without_dynamic_gcn'] == 1:
            from DyMF.model_without_dynamic_gcn import Encoder, Decoder
            from DyMF.runner import train
        elif args['without_tactical_fusion'] == 1:
            from DyMF.model_without_tactical_fusion import Encoder, Decoder
            from DyMF.runner import train
        elif args['without_player_style_fusion'] == 1:
            from DyMF.model_without_player_style_fusion import Encoder, Decoder
            from DyMF.runner import train
        elif args['without_rally_fusion'] == 1:
            from DyMF.model_without_rally_fusion import Encoder, Decoder
            from DyMF.runner import train
        elif args['without_style_fusion'] == 1:
            from DyMF.model_without_style_fusion import Encoder, Decoder
            from DyMF.runner import train
        else:
            from DyMF.model import Encoder, Decoder
            from DyMF.runner import train

        encoder = Encoder(args, device)
        decoder = Decoder(args, device)
        encoder.player_embedding.weight = decoder.player_embedding.weight
        encoder.coordination_transform.weight = decoder.coordination_transform.weight
        # encoder.rGCN.type_embedding.weight = decoder.rGCN.type_embedding.weight
    
    if args['model_type'] == 'GCN':
        if args['use_complete_graph'] == 1:
            from GCN.model_complete import Encoder, Decoder
        else:
            from GCN.model import Encoder, Decoder
        from GCN.runner import train
        encoder = Encoder(args)
        decoder = Decoder(args)
        encoder.player_embedding.weight = decoder.player_embedding.weight
        encoder.coordination_transform.weight = decoder.coordination_transform.weight

    if args['model_type'] == 'ShuttleNet':
        from ShuttleNet.ShuttleNet import ShotGenEncoder, ShotGenPredictor
        from ShuttleNet.runner import train
        encoder = ShotGenEncoder(args)
        decoder = ShotGenPredictor(args)
        encoder.player_embedding.weight = decoder.shotgen_decoder.player_embedding.weight
        encoder.type_embedding.weight = decoder.shotgen_decoder.type_embedding.weight
        encoder.coordination_transform.weight = decoder.shotgen_decoder.coordination_transform.weight

    if args['model_type'] == 'rGCN':
        if args['use_complete_graph'] == 1:
            from rGCN.model_complete import Encoder, Decoder
        else:
            from rGCN.model import Encoder, Decoder
        from rGCN.runner import train
        encoder = Encoder(args, device)
        decoder = Decoder(args, device)
        encoder.player_embedding.weight = decoder.player_embedding.weight
        encoder.coordination_transform.weight = decoder.coordination_transform.weight

    if args['model_type'] == 'Transformer':
        from Transformer.transformer import TransformerEncoder, TransformerPredictor
        from Transformer.runner import train
        encoder = TransformerEncoder(args)
        decoder = TransformerPredictor(args)
        encoder.player_embedding.weight = decoder.transformer_decoder.player_embedding.weight
        encoder.type_embedding.weight = decoder.transformer_decoder.type_embedding.weight
        encoder.coordination_transform.weight = decoder.transformer_decoder.coordination_transform.weight

    if args['model_type'] == 'GCN_d':
        if args['use_complete_graph'] == 1:
            from GCN_dynamic.model_complete import Encoder, Decoder
        else:
            from GCN_dynamic.model import Encoder, Decoder
        from GCN_dynamic.runner import train
        encoder = Encoder(args, device)
        decoder = Decoder(args, device)
        encoder.player_embedding.weight = decoder.player_embedding.weight
        encoder.coordination_transform.weight = decoder.coordination_transform.weight

    if args['model_type'] == 'eGCN':
        if args['use_complete_graph'] == 1:
            from eGCN.model_complete import Encoder, Decoder
        else:
            from eGCN.model import Encoder, Decoder
        from eGCN.runner import train
        encoder = Encoder(args)
        decoder = Decoder(args)
        encoder.player_embedding.weight = decoder.player_embedding.weight
        encoder.coordination_transform.weight = decoder.coordination_transform.weight

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args['lr'])
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args['lr'])

    location_criterion = nn.MSELoss()
    shot_type_criterion = nn.CrossEntropyLoss()

    encoder.to(device), decoder.to(device), location_criterion.to(device), shot_type_criterion.to(device)

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(total_params)

    train_loss, train_loss_location, train_loss_type = train(train_dataloader, valid_dataloader, encoder, decoder, location_criterion, shot_type_criterion, encoder_optimizer, decoder_optimizer, args, device=device)
    save_args_file(args)

    print(args['model_folder'])
    print("total loss: {}".format(train_loss))
    print("location loss: {}".format(train_loss_location))
    print("type loss: {}".format(train_loss_type))

if __name__ == '__main__':
    main()