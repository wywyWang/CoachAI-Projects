from badmintoncleaner import prepare_dataset
import ast
import sys
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)    # gpu vars


if __name__ == "__main__":
    SAMPLES = 6                               # set to 6 to meet the requirement of this challenge

    model_path = sys.argv[1]
    config = ast.literal_eval(open(f"{model_path}/config").readline())
    set_seed(config['seed_value'])

    # Prepare Dataset
    config, train_dataloader, val_dataloader, test_dataloader, train_matches, val_matches, test_matches = prepare_dataset(config)
    device = torch.device(f"cuda:{config['gpu_num']}" if torch.cuda.is_available() else "cpu")

    # load model
    from ShuttleNet.ShuttleNet import ShotGenEncoder, ShotGenPredictor
    from ShuttleNet.ShuttleNet_runner import shotgen_generator
    encoder = ShotGenEncoder(config)
    decoder = ShotGenPredictor(config)

    encoder.to(device), decoder.to(device)
    current_model_path = model_path + '/'
    encoder_path = current_model_path + 'encoder'
    decoder_path = current_model_path + 'decoder'
    encoder.load_state_dict(torch.load(encoder_path, map_location=device)), decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    encode_length = config['encode_length']

    performance_log = open(f"{current_model_path}prediction.csv", "a")
    performance_log.write('rally_id,sample_id,ball_round,landing_x,landing_y,short service,net shot,lob,clear,drop,push/rush,smash,defensive shot,drive,long service')
    performance_log.write('\n')

    # get all testing rallies
    testing_rallies = val_matches['rally_id'].unique()

    for rally_id in tqdm(testing_rallies):
        # read data
        selected_matches = val_matches.loc[(val_matches['rally_id'] == rally_id)][['rally_id', 'type', 'landing_x', 'landing_y', 'player', 'rally_length']].reset_index(drop=True)
        
        generated_length = selected_matches['rally_length'][0]      # fetch the length of the current rally
        players = [selected_matches['player'][0], selected_matches['player'][1]]
        target_players = torch.tensor([players[shot_index%2] for shot_index in range(generated_length-len(selected_matches))])  # get the predicted players
        
        given_seq = {
            'given_player': torch.tensor(selected_matches['player'].values).to(device),
            'given_shot': torch.tensor(selected_matches['type'].values).to(device),
            'given_x': torch.tensor(selected_matches['landing_x'].values).to(device),
            'given_y': torch.tensor(selected_matches['landing_y'].values).to(device),
            'target_player': target_players.to(device),
            'rally_length': generated_length
        }

        # feed into the model
        generated_shot, generated_area = shotgen_generator(given_seq=given_seq, encoder=encoder, decoder=decoder, config=config, samples=SAMPLES, device=device)

        # store the prediction results
        for sample_id in range(len(generated_area)):
            for ball_round in range(len(generated_area[0])):
                performance_log.write(f"{rally_id},{sample_id},{ball_round+config['encode_length']+1},{generated_area[sample_id][ball_round][0]},{generated_area[sample_id][ball_round][1]},")
                for shot_id, shot_type_logits in enumerate(generated_shot[sample_id][ball_round]):
                    performance_log.write(f"{shot_type_logits}")
                    if shot_id != len(generated_shot[sample_id][ball_round]) - 1:
                        performance_log.write(",")
                performance_log.write("\n")
