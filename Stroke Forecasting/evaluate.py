from badmintoncleaner import prepare_dataset
from utils import evaluation_rnn, evaluation_non_rnn
import ast
import sys
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn


def draw_record(record, save_path):
    for key, value in record.items():
        fig = plt.figure(figsize=(12, 6))
        ball_round = np.arange(len(record[key]))
        plt.title("{} loss".format(key))
        plt.xlabel("Ball round")
        plt.ylabel("Loss")
        plt.grid()
        plt.bar(ball_round, record[key])
        plt.savefig('{}{}_bar.png'.format(save_path, key))
        plt.close(fig)


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)    # gpu vars


model_path = sys.argv[1]
config = ast.literal_eval(open(model_path + '1/' + 'config').readline())
SAMPLES = config['sample']
set_seed(config['seed_value'])

# Prepare Dataset
matches, total_train, total_val, total_test, config = prepare_dataset(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = {
    'entropy': nn.CrossEntropyLoss(ignore_index=0, reduction='sum'),
    'mse': nn.MSELoss(reduction='sum'),
    'mae': nn.L1Loss(reduction='sum')
}
for key, value in criterion.items():
    criterion[key].to(device)

rnn_model = ['LSTM', 'CFLSTM']

k_fold_index = 0
total_test_loss = {
    'entropy': [],
    'mse': [],
    'mae': []
}
result_log = open(model_path + 'result.log', 'w')
performance_log = open('{}_result_{}_{}.log'.format(config['model_type'], str(config['encode_length']), str(config['sample'])), 'a')
for train_dataloader, test_dataloader in zip(total_train, total_test):
    k_fold_index += 1

    # load model
    if config['model_type'] == 'LSTM':
        from LSTM.GRU import GRUEncoder, GRUDecoder
        encoder = GRUEncoder(config)
        decoder = GRUDecoder(config)
    elif config['model_type'] == 'Transformer':
        from Transformer.transformer import TransformerEncoder, TransformerPredictor
        encoder = TransformerEncoder(config)
        decoder = TransformerPredictor(config)
    elif config['model_type'] == 'CFLSTM':
        from CFLSTM.cf_lstm import CFLSTMEncoder, CFLSTMDecoder
        encoder = CFLSTMEncoder(config)
        decoder = CFLSTMDecoder(config)
    elif config['model_type'] == 'DNRI':
        from DNRI.DNRI import DNRIEncoder, DNRIDecoder
        from DNRI.dnri_runner import DNRI_evaluate_perplexity_mse
        encoder = DNRIEncoder(config)
        decoder = DNRIDecoder(config)
    elif config['model_type'] == 'DMA_Nets':
        from DMA_Nets.DMA_Nets import DMA_Nets_Encoder, DMA_Nets_Decoder
        from DMA_Nets.dma_nets_runner import DMA_Nets_evaluate_perplexity_mse
        encoder = DMA_Nets_Encoder(config)
        decoder = DMA_Nets_Decoder(config)
    elif config['model_type'] == 'ShuttleNet':
        from ShuttleNet.ShuttleNet import ShotGenEncoder, ShotGenPredictor
        encoder = ShotGenEncoder(config)
        decoder = ShotGenPredictor(config)
    elif config['model_type'] == 'ours_rm_taa':
        from ours_rm_taa.shotGen import ShotGenEncoder, ShotGenPredictor
        encoder = ShotGenEncoder(config)
        decoder = ShotGenPredictor(config)
    elif config['model_type'] == 'ours_p2r':
        from ours_p2r.shotGen_hie import ShotGenEncoder_hie, ShotGenPredictor_hie
        encoder = ShotGenEncoder_hie(config)
        decoder = ShotGenPredictor_hie(config)
    elif config['model_type'] == 'ours_r2p':
        from ours_r2p.shotGen_hie import ShotGenEncoder_hie, ShotGenPredictor_hie
        encoder = ShotGenEncoder_hie(config)
        decoder = ShotGenPredictor_hie(config)
    else:
        raise NotImplementedError

    encoder.to(device), decoder.to(device)
    current_model_path = model_path + str(k_fold_index) + '/'
    encoder_path = current_model_path + 'encoder'
    decoder_path = current_model_path + 'decoder'
    encoder.load_state_dict(torch.load(encoder_path)), decoder.load_state_dict(torch.load(decoder_path))

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print("Model params: {}".format(total_params))

    # run evaluation
    if config['model_type'] in rnn_model:
        test_loss, test_record = evaluation_rnn(test_dataloader, encoder, decoder, criterion, config, config['model_type'], samples=SAMPLES, device=device)
    elif config['model_type'] == 'DNRI':
        test_loss = DNRI_evaluate_perplexity_mse(test_dataloader, encoder, decoder, criterion, config, samples=SAMPLES, device=device)
    elif config['model_type'] == 'DMA_Nets':
        test_loss = DMA_Nets_evaluate_perplexity_mse(test_dataloader, encoder, decoder, criterion, config, samples=SAMPLES, device=device)
    else:
        test_loss, test_record = evaluation_non_rnn(test_dataloader, encoder, decoder, criterion, config, config['model_type'], samples=SAMPLES, device=device)

    for key, value in total_test_loss.items():
        total_test_loss[key].append(test_loss[key])
    
    # write log
    result_log.write(str(k_fold_index))
    result_log.write('\n')
    result_log.write("[Test] [Instance: {}] [Loss: {}] [Shot CE loss: {}] [Area MSE loss: {}] [Area MAE loss: {}]".format(test_loss['count'], test_loss['total'], test_loss['entropy'], test_loss['mse'], test_loss['mae']))
    result_log.write('\n')
    if config['model_type'] != 'DNRI' and config['model_type'] != 'DMA_Nets':
        result_log.write(str(test_record))

    performance_log.write(current_model_path + "," + str(test_loss['total']) + "," + str(test_loss['entropy']) + "," + str(test_loss['mse']) + "," + str(test_loss['mae']))
    performance_log.write('\n')

    print("[Test] [Instance: {}] [Loss: {}] [Shot CE loss: {}] [Area MSE loss: {}] [Area MAE loss: {}]".format(test_loss['count'], test_loss['total'], test_loss['entropy'], test_loss['mse'], test_loss['mae']))

    if config['model_type'] != 'DNRI' and config['model_type'] != 'DMA_Nets':
        draw_record(test_record, current_model_path)

# # used for k-fold hyperparameters tuning
# import statistics
# for key, value in total_test_loss.items():
#     mean = round(statistics.mean(total_test_loss[key]), 4)
#     std = round(statistics.stdev(total_test_loss[key]), 4)
#     print("[Test {}] [{} +- {}]".format(key, mean, std))
#     result_log.write("[Test {}] [Mean: {}] [Std: {}]".format(key, mean, std))