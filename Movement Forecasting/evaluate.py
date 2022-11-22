import sys
import torch
import torch.nn as nn
import random
import numpy as np

from prepare_dataset import prepare_dataset
from utils import load_args_file

import os

model_folder = sys.argv[1]
sample_num = sys.argv[2]

args = load_args_file(model_folder)
args['sample_num'] = int(sample_num)

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

train_dataloader, valid_dataloader, test_dataloader, args = prepare_dataset(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args['model_type'] == 'DNRI':
    from DNRI.model import Encoder, Decoder
    from DNRI.runner import evaluate
    encoder = Encoder(args)
    decoder = Decoder(args)
    encoder.player_embedding.weight = decoder.player_embedding.weight
    encoder.coordination_transform.weight = decoder.coordination_transform.weight

if args['model_type'] == 'LSTM':
    from LSTM.model import Encoder, Decoder
    from LSTM.runner import evaluate
    encoder = Encoder(args)
    decoder = Decoder(args)
    encoder.player_embedding.weight = decoder.player_embedding.weight
    encoder.type_embedding.weight = decoder.type_embedding.weight
    encoder.coordination_transform.weight = decoder.coordination_transform.weight

if args['model_type'] == 'DyMF':
    if args['use_complete_graph'] == 1:
        from DyMF.model_complete import Encoder, Decoder
        from DyMF.runner import evaluate
    elif args['without_dynamic_gcn'] == 1:
        from DyMF.model_without_dynamic_gcn import Encoder, Decoder
        from DyMF.runner import evaluate
    elif args['without_tactical_fusion'] == 1:
        from DyMF.model_without_tactical_fusion import Encoder, Decoder
        from DyMF.runner import evaluate
    elif args['without_player_style_fusion'] == 1:
        from DyMF.model_without_player_style_fusion import Encoder, Decoder
        from DyMF.runner import evaluate
    elif args['without_rally_fusion'] == 1:
        from DyMF.model_without_rally_fusion import Encoder, Decoder
        from DyMF.runner import evaluate
    elif args['without_style_fusion'] == 1:
        from DyMF.model_without_style_fusion import Encoder, Decoder
        from DyMF.runner import evaluate
    else:
        from DyMF.model import Encoder, Decoder
        from DyMF.runner import evaluate

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
    from GCN.runner import evaluate
    encoder = Encoder(args)
    decoder = Decoder(args)
    encoder.player_embedding.weight = decoder.player_embedding.weight
    encoder.coordination_transform.weight = decoder.coordination_transform.weight

if args['model_type'] == 'ShuttleNet':
    from ShuttleNet.ShuttleNet import ShotGenEncoder, ShotGenPredictor
    from ShuttleNet.runner import evaluate
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
    from rGCN.runner import evaluate
    encoder = Encoder(args, device)
    decoder = Decoder(args, device)
    encoder.player_embedding.weight = decoder.player_embedding.weight
    encoder.coordination_transform.weight = decoder.coordination_transform.weight

if args['model_type'] == 'Transformer':
    from Transformer.transformer import TransformerEncoder, TransformerPredictor
    from Transformer.runner import evaluate
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
    from GCN_dynamic.runner import evaluate
    encoder = Encoder(args, device)
    decoder = Decoder(args, device)
    encoder.player_embedding.weight = decoder.player_embedding.weight
    encoder.coordination_transform.weight = decoder.coordination_transform.weight

if args['model_type'] == 'eGCN':
    if args['use_complete_graph'] == 1:
        from eGCN.model_complete import Encoder, Decoder
    else:
        from eGCN.model import Encoder, Decoder
    from eGCN.runner import evaluate
    encoder = Encoder(args)
    decoder = Decoder(args)
    encoder.player_embedding.weight = decoder.player_embedding.weight
    encoder.coordination_transform.weight = decoder.coordination_transform.weight

location_MSE_criterion = nn.MSELoss(reduction='sum')
location_MAE_criterion = nn.L1Loss(reduction='sum')
shot_type_criterion = nn.CrossEntropyLoss()


encoder.load_state_dict(torch.load(args['model_folder'] + '/encoder')), decoder.load_state_dict(torch.load(args['model_folder'] + '/decoder'))

encoder.to(device), decoder.to(device), location_MSE_criterion.to(device), location_MAE_criterion.to(device), shot_type_criterion.to(device)

test_loss, test_loss_MSE_location, test_loss_MAE_location, test_loss_type = evaluate(test_dataloader, encoder, decoder, location_MSE_criterion, location_MAE_criterion, shot_type_criterion, args, device=device)
print("total loss: {:.4f}".format(test_loss))
print("location MSE loss: {:.4f}".format(test_loss_MSE_location))
print("location MAE loss: {:.4f}".format(test_loss_MAE_location))
print("type loss: {:.4f}".format(test_loss_type))