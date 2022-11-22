from badmintoncleaner import prepare_dataset
import ast
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# y應該要倒過來，就倒過來看，原本下半場站的人現在在上半
def draw(save_path, predict_area, predict_shot, true_area, true_shot):
    # https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
    fig = plt.figure(dpi=200)
    plt.axis('off')

    shot_type_transform = \
    {
        '發短球': 'short service',
        '長球': 'clear',
        '推撲球': 'push/rush',
        '殺球': 'smash',
        '接殺防守': 'defensive shot',
        '平球': 'drive',
        '網前球': 'net shot',
        '挑球': 'lob',
        '切球': 'drop',
        '發長球': 'long service',
    }

    # draw predict points
    x, y, colors = [], [], []
    for i in range(len(predict_area)):
        i_x = predict_area[i][0]
        i_y = predict_area[i][1]
        type_chinese = predict_shot[i]
        x.append(i_x)
        y.append(i_y)

        if (i+1+5) % 2 == 0:
            colors.append('r')
        else:
            colors.append('b')

        if i+1+5 == 8:
            plt.annotate('{}: {}'.format(i+1+5, shot_type_transform[type_chinese]), (i_x + 15, i_y - 20), rotation=90, fontsize=12)
        else:
            plt.annotate('{}: {}'.format(i+1+5, shot_type_transform[type_chinese]), (i_x + 5, i_y - 20), rotation=90, fontsize=12)
    plt.scatter(x, y, c=colors, marker='o', label='predicted shots')

    # draw ground truth points
    x_given, y_given, x_true, y_true, colors_given, colors_true = [], [], [], [], [], []
    for i in range(len(true_area)):
        i_x = true_area[i][0]
        i_y = true_area[i][1]
        i_shot = true_shot[i]
        type_chinese = config['uniques_type'][i_shot-1]

        if i < 5:
            colors_given.append('gray')
            x_given.append(i_x)
            y_given.append(i_y)
        else:
            x_true.append(i_x)
            y_true.append(i_y)
            if (i+1) % 2 == 0:
                colors_true.append('r')
            else:
                colors_true.append('b')
        if i+1 == 1:
            plt.annotate('{}: {}'.format(i+1, shot_type_transform[type_chinese]), (i_x + 5, i_y - 20), rotation=90, fontsize=12)
        elif i+1 == 8:
            plt.annotate('{}: {}'.format(i+1, shot_type_transform[type_chinese]), (i_x + 5, i_y - 50), rotation=90, fontsize=12)
        elif i+1 == 4:
            plt.annotate('{}: {}'.format(i+1, shot_type_transform[type_chinese]), (i_x - 15, i_y - 60), rotation=90, fontsize=12)
        else:
            plt.annotate('{}: {}'.format(i+1, shot_type_transform[type_chinese]), (i_x + 5, i_y - 40), rotation=90, fontsize=12)
    plt.scatter(x_given, y_given, c=colors_given, marker='^', label='past shots')
    plt.scatter(x_true, y_true, c=colors_true, marker='1', label='true shots')

    # draw court line (50, 810)
    # draw vertical
    plt.plot((25, 25), (150, 810), c='gray')       # court line, not single line
    plt.plot((330, 330), (150, 810), c='gray')       # court line, not single line
    plt.plot((305, 305), (150, 810), c='black')
    plt.plot((50, 50), (150, 810), c='black')
    plt.plot((177.5, 177.5), (150, 810), c='black')

    # draw horizontal
    plt.plot((25, 330), (480, 480), c='red')
    plt.plot((50, 305), (150, 150), c='black')
    plt.plot((50, 305), (810, 810), c='black')
    plt.plot((50, 305), (204, 204), c='black')
    plt.plot((50, 305), (756, 756), c='black')
    plt.plot((25, 50), (150, 150), c='gray')
    plt.plot((305, 330), (150, 150), c='gray')
    plt.plot((25, 50), (810, 810), c='gray')
    plt.plot((305, 330), (810, 810), c='gray')
    plt.plot((25, 50), (204, 204), c='gray')
    plt.plot((305, 330), (204, 204), c='gray')
    plt.plot((25, 50), (756, 756), c='gray')
    plt.plot((305, 330), (756, 756), c='gray')

    plt.gcf().set_size_inches(6, 8)
    # plt.legend(bbox_to_anchor=(0, 1, 0.95, 0.08), borderaxespad=0.)
    plt.savefig(save_path)
    plt.close(fig)


def visualize_attention(attention, mode):
    if mode == 'decoder':
        prefix = 'attention_weight/{}_{}'.format(mode, config['model_type'])
    elif mode == 'encoder_decoder':
        prefix = 'attention_weight/{}_{}'.format(mode, config['model_type'])
    elif mode == 'disentangled':
        prefix = 'attention_weight/{}_{}'.format(mode, config['model_type'])
    else:
        raise NotImplementedError

    if mode != 'disentangled':
        fig, axs = plt.subplots(ncols=4, figsize=(14, 8), gridspec_kw=dict(width_ratios=[8, 8, 8, 1]))
        for head in range(attention.shape[1]):
            axs[head].set_title('{} Head {}'.format(config['model_type'], head))
            sns_plot = sns.heatmap(attention[0, head], vmin=0, vmax=1, ax=axs[head], cbar=False, annot=False, linewidths=0.0, cmap="viridis")

        axs[2].set_title('{} Mean'.format(config['model_type']))
        sns_plot = sns.heatmap(np.mean(attention[0, :], axis=0), vmin=0, vmax=1, ax=axs[2], cbar=False, annot=False, linewidths=0.0, cmap="viridis")
        
        fig.colorbar(axs[1].collections[0], cax=axs[3])
        plt.tight_layout()
        sns_plot.figure.savefig('{}_mean'.format(prefix))
    else:
        fig, axs = plt.subplots(ncols=5, figsize=(16, 8), gridspec_kw=dict(width_ratios=[8, 8, 8, 8, 1]))
        for idx, key in enumerate(attention.keys()):
            each_attention = np.mean(attention[key].cpu().detach().numpy()[0, :], axis=0)
            axs[idx].set_title('{} {}'.format(config['model_type'], key))
            sns_plot = sns.heatmap(each_attention, vmin=0, vmax=1, ax=axs[idx], cbar=False, annot=False, linewidths=0.0, cmap="viridis")
        
        fig.colorbar(axs[1].collections[0], cax=axs[-1])
        plt.tight_layout()
        sns_plot.figure.savefig('{}_mean'.format(prefix))


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)    # gpu vars


SAMPLES = 10

model_path = sys.argv[1]
config = ast.literal_eval(open(model_path + '1/' + 'config').readline())

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


# load model
if config['model_type'] == 'LSTM':
    from LSTM.GRU import GRUEncoder, GRUDecoder
    from LSTM.gru_runner import GRU_generator
    encoder = GRUEncoder(config)
    decoder = GRUDecoder(config)
elif config['model_type'] == 'Transformer':
    from Transformer.transformer import TransformerEncoder, TransformerPredictor
    from Transformer.transformer_runner import transformer_generator
    encoder = TransformerEncoder(config)
    decoder = TransformerPredictor(config)
elif config['model_type'] == 'DMA_Nets':
    from DMA_Nets.DMA_Nets import DMA_Nets_Encoder, DMA_Nets_Decoder
    from DMA_Nets.dma_nets_runner import DMA_Nets_generator
    encoder = DMA_Nets_Encoder(config)
    decoder = DMA_Nets_Decoder(config)
elif config['model_type'] == 'CFLSTM':
    from CFLSTM.cf_lstm import CFLSTMEncoder, CFLSTMDecoder
    from CFLSTM.cf_lstm_runner import CFLSTM_generator
    encoder = CFLSTMEncoder(config)
    decoder = CFLSTMDecoder(config)
elif config['model_type'] == 'FuRaptor':
    from FuRaptor.FuRaptor import ShotGenEncoder, ShotGenPredictor
    from FuRaptor.FuRaptor_runner import shotgen_generator
    encoder = ShotGenEncoder(config)
    decoder = ShotGenPredictor(config)
elif config['model_type'] == 'DNRI':
    from DNRI.DNRI import DNRIEncoder, DNRIDecoder
    from DNRI.dnri_runner import DNRI_generator
    encoder = DNRIEncoder(config)
    decoder = DNRIDecoder(config)
else:
    raise NotImplementedError


criterion = {
    'entropy': nn.CrossEntropyLoss(ignore_index=0, reduction='sum'),
    'mse': nn.MSELoss(reduction='sum'),
    'mae': nn.L1Loss(reduction='sum')
}
for key, value in criterion.items():
    criterion[key].to(device)

encoder.to(device), decoder.to(device)
current_model_path = model_path + '1/'
encoder_path = current_model_path + 'encoder'
decoder_path = current_model_path + 'decoder'
encoder.load_state_dict(torch.load(encoder_path)), decoder.load_state_dict(torch.load(decoder_path))

encode_length = config['encode_length']
max_ball_round = config['max_ball_round']

performance_log = open('log/{}_result_{}.log'.format(config['model_type'], str(config['encode_length'])), 'a')
performance_log.write('rally_id,total_instance,total,ce,mse,mae')
performance_log.write('\n')

total_loss = {
    'rally_id': [],
    'total': [],
    'entropy': [],
    'mse': [],
    'mae': []
}

for rally_id in tqdm(range(matches['rally_id'].nunique())):
# for rally_id in [421]:
    selected_matches = matches.loc[(matches['rally_id'] == rally_id)][['rally_id', 'type', 'landing_x', 'landing_y', 'player']].reset_index(drop=True)
    
    # for case study
    # selected_matches[selected_matches['player'] == 2] = 3
    # selected_matches[selected_matches['player'] == 1] = 2
    # selected_matches[selected_matches['player'] == 3] = 1

    if len(selected_matches) <= (encode_length+1):
        continue

    if len(selected_matches['player']) <= max_ball_round:
        given_seq = {
            'given_player': torch.tensor(selected_matches['player'][:-1].values).to(device),
            'given_shot': torch.tensor(selected_matches['type'][:-1].values).to(device),
            'given_x': torch.tensor(selected_matches['landing_x'][:-1].values).to(device),
            'given_y': torch.tensor(selected_matches['landing_y'][:-1].values).to(device),
            'target_player': torch.tensor(selected_matches['player'][1:].values).to(device)
        }

        correct_seq = {
            'given_shot': torch.tensor(selected_matches['type'][1:].values).to(device),
            'given_x': torch.tensor(selected_matches['landing_x'][1:].values).to(device),
            'given_y': torch.tensor(selected_matches['landing_y'][1:].values).to(device)
        }
    else:
        given_seq = {
            'given_player': torch.tensor(selected_matches['player'][:max_ball_round].values).to(device),
            'given_shot': torch.tensor(selected_matches['type'][:max_ball_round].values).to(device),
            'given_x': torch.tensor(selected_matches['landing_x'][:max_ball_round].values).to(device),
            'given_y': torch.tensor(selected_matches['landing_y'][:max_ball_round].values).to(device),
            'target_player': torch.tensor(selected_matches['player'][1:].values).to(device)
        }

        correct_seq = {
            'given_shot': torch.tensor(selected_matches['type'][1:].values).to(device),
            'given_x': torch.tensor(selected_matches['landing_x'][1:].values).to(device),
            'given_y': torch.tensor(selected_matches['landing_y'][1:].values).to(device)
        }

    # print(selected_matches)
    # print(len(given_seq['given_player']))
    # print(len(given_seq['given_x']))
    # print(len(given_seq['given_y']))
    # print(len(given_seq['target_player']))
    # 1/0

    if config['model_type'] == 'LSTM':
        current_loss, total_instance, generated_area, generated_shot = GRU_generator(given_seq, correct_seq, encoder, decoder, criterion, config, SAMPLES, device)
    elif config['model_type'] == 'CFLSTM':
        current_loss, total_instance, generated_area, generated_shot = CFLSTM_generator(given_seq, correct_seq, encoder, decoder, criterion, config, SAMPLES, device)
    elif config['model_type'] == 'Transformer':
        current_loss, total_instance, generated_area, generated_shot, generated_decoder_attention, generated_decoder_encoder_attention = transformer_generator(given_seq, correct_seq, encoder, decoder, criterion, config, SAMPLES, device)
    elif config['model_type'] == 'DMA_Nets':
        current_loss, total_instance, generated_area, generated_shot = DMA_Nets_generator(given_seq, correct_seq, encoder, decoder, criterion, config, SAMPLES, device)
    elif config['model_type'] == 'FuRaptor':
        current_loss, total_instance, generated_area, generated_shot, generated_decoder_attention, generated_decoder_encoder_attention, generated_decoder_encoder_disentangled = shotgen_generator(given_seq, correct_seq, encoder, decoder, criterion, config, SAMPLES, device)
    elif config['model_type'] == 'DNRI':
        current_loss, total_instance, generated_area, generated_shot = DNRI_generator(given_seq, correct_seq, encoder, decoder, criterion, config, SAMPLES, device)
    else:
        raise NotImplementedError

    total = round(current_loss['total'].item(), 3)
    ce = round(current_loss['entropy'].item(), 3)
    mse = round(current_loss['mse'].item(), 3)
    mae = round(current_loss['mae'].item(), 3)

    performance_log.write(str(rally_id) + "," + str(total_instance) + "," + str(total) + "," + str(ce) + "," + str(mse) + "," + str(mae))
    performance_log.write('\n')

    mean_x, std_x = 175., 82.
    mean_y, std_y = 467., 192.
    true_area = list(zip(selected_matches['landing_x'].values * std_x + mean_x, selected_matches['landing_y'].values * std_y + mean_y))
    # true_area = list(zip(selected_matches['landing_x'].values, selected_matches['landing_y'].values))
    true_shot = selected_matches['type'].values

    save_path = 'print_results/print_{}/rally_{}.png'.format(config['model_type'], rally_id)
    draw(save_path, generated_area, generated_shot, true_area, true_shot)


# # rally=13, rally= 14 is long rally
# selected_rally = 851
# selected_matches = matches.loc[(matches['rally_id'] == selected_rally)][['rally_id', 'type', 'landing_x', 'landing_y', 'player']].reset_index(drop=True)

# given_seq = {
#     'given_player': torch.tensor(selected_matches['player'][:-1].values).to(device),
#     'given_shot': torch.tensor(selected_matches['type'][:-1].values).to(device),
#     'given_x': torch.tensor(selected_matches['landing_x'][:-1].values).to(device),
#     'given_y': torch.tensor(selected_matches['landing_y'][:-1].values).to(device),
#     'target_player': torch.tensor(selected_matches['player'][1:].values).to(device)
# }

# correct_seq = {
#     'given_shot': torch.tensor(selected_matches['type'][1:].values).to(device),
#     'given_x': torch.tensor(selected_matches['landing_x'][1:].values).to(device),
#     'given_y': torch.tensor(selected_matches['landing_y'][1:].values).to(device)
# }

# # print(selected_matches)
# # print(len(given_seq['given_player']))
# # print(len(given_seq['given_x']))
# # print(len(given_seq['given_y']))
# # print(len(given_seq['target_player']))
# # 1/0

# if config['model_type'] == 'LSTM':
#     current_loss, generated_area, generated_shot = GRU_generator(given_seq, correct_seq, encoder, decoder, criterion, config, SAMPLES, device)
# elif config['model_type'] == 'CFLSTM':
#     current_loss, generated_area, generated_shot = CFLSTM_generator(given_seq, correct_seq, encoder, decoder, criterion, config, SAMPLES, device)
# elif config['model_type'] == 'Transformer':
#     current_loss, generated_area, generated_shot, generated_decoder_attention, generated_decoder_encoder_attention = transformer_generator(given_seq, correct_seq, encoder, decoder, criterion, config, SAMPLES, device)
# elif config['model_type'] == 'ShotGen' or config['model_type'] == 'ShotGen_v2':
#     current_loss, total_instance, generated_area, generated_shot, generated_decoder_attention, generated_decoder_encoder_attention, generated_decoder_encoder_disentangled = shotgen_generator(given_seq, correct_seq, encoder, decoder, criterion, config, SAMPLES, device)
# elif config['model_type'] == 'DNRI':
#     current_loss, generated_area, generated_shot = DNRI_generator(given_seq, correct_seq, encoder, decoder, criterion, config, SAMPLES, device)
# else:
#     raise NotImplementedError

# mean_x, std_x = 175., 82.
# mean_y, std_y = 467., 192.
# true_area = list(zip(selected_matches['landing_x'].values * std_x + mean_x, selected_matches['landing_y'].values * std_y + mean_y))
# # true_area = list(zip(selected_matches['landing_x'].values, selected_matches['landing_y'].values))
# true_shot = selected_matches['type'][5:].values

# for area, shot in zip(true_area, true_shot):
#     print(area, config['uniques_type'][shot-1])
# print()
# for area, shot in zip(generated_area, generated_shot):
#     print(area, shot)

# print(config['uniques_type'])

# if config['model_type'] == 'ShotGen':
#     visualize_attention(generated_decoder_encoder_disentangled, mode='disentangled')
# visualize_attention(generated_decoder_attention, mode='decoder')
# visualize_attention(generated_decoder_encoder_attention, mode='encoder_decoder')

# save_path = 'attention_weight/{}_{}.png'.format(config['model_type'], selected_rally)
# draw(save_path, generated_area, true_area)
