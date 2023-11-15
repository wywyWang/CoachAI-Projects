from badmintoncleaner import prepare_dataset
from utils import evaluation
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
        plt.title(f"{key} loss")
        plt.xlabel("Ball round")
        plt.ylabel("Loss")
        plt.grid()
        plt.bar(ball_round, record[key])
        plt.savefig(f'{save_path}{key}_bar.png')
        plt.close(fig)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # gpu vars

def main():
    #model_path = sys.argv[1]
    model_path = "./StrokeForecasting/evaluate_3GMM90"
    config = ast.literal_eval(open(f"{model_path}1/config",encoding='utf8').readline())
    SAMPLES = config['sample']
    set_seed(config['seed_value'])

    # Prepare Dataset
    matches, total_train, total_val, total_test, config = prepare_dataset(config)
    print(f"palyer count{config['player_num']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = {
        'entropy': nn.CrossEntropyLoss(ignore_index=0, reduction='sum'),
        'mse': nn.MSELoss(reduction='sum'),
        'mae': nn.L1Loss(reduction='sum')
    }
    for key, value in criterion.items():
        criterion[key].to(device)

    k_fold_index = 0
    total_test_loss = {
        'entropy': [],
        'mse': [],
        'mae': []
    }
    result_log = open(model_path + 'result.log', 'w')
    performance_log = open(f"SuttleNet_result_{config['encode_length']}_{config['sample']}.log", 'a')
    for train_dataloader, test_dataloader in zip(total_train, total_test):
        k_fold_index += 1

        # load model
        from ShuttleNet.Decoder import ShotGenDecoder
        from ShuttleNet.Encoder import ShotGenEncoder
        encoder = ShotGenEncoder(config)
        decoder = ShotGenDecoder(config)

        encoder.to(device), decoder.to(device)
        current_model_path = f"{model_path}{k_fold_index}/"
        encoder_path = f"{current_model_path}encoder"
        decoder_path = f"{current_model_path}decoder"
        encoder.load_state_dict(torch.load(encoder_path)), decoder.load_state_dict(torch.load(decoder_path))

        total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) \
                     + sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        print(f"Model params: {total_params}")

        # run evaluation
        test_loss, test_record = evaluation(test_dataloader, encoder, decoder, criterion, config, samples=SAMPLES, device=device)

        for key, value in total_test_loss.items():
            total_test_loss[key].append(test_loss[key])

        # write log
        result_log.write(f"{k_fold_index}\n")
        result = "[Test] [Instance: {count}] [Loss: {total}] [Shot CE loss: {entropy}] [Area MSE loss: {mse}] [Area MAE loss: {mae}]\n".format(**test_loss)
        result_log.write(result)
        result_log.write(str(test_record))

        performance_log.write(f"{current_model_path},{test_loss['total']},{test_loss['entropy']},{test_loss['mse']},{test_loss['mae']}\n")

        print(result)

        draw_record(test_record, current_model_path)

    # # used for k-fold hyperparameters tuning
    # import statistics
    # for key, value in total_test_loss.items():
    #     mean = round(statistics.mean(total_test_loss[key]), 4)
    #     std = round(statistics.stdev(total_test_loss[key]), 4)
    #     print("[Test {}] [{} +- {}]".format(key, mean, std))
    #     result_log.write("[Test {}] [Mean: {}] [Std: {}]".format(key, mean, std))

if __name__ == "__main__":
    main()