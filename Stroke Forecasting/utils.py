import os
import torch
import torch.nn.functional as F
import torch.distributions.multivariate_normal as torchdist
from tqdm import tqdm


def evaluation_non_rnn(data_loader, encoder, decoder, criterion, config, model_type, samples=1, device="cpu"):
    encode_length = config['encode_length']
    encoder.eval(), decoder.eval()
    total_loss = {
        'total': 0,
        'entropy': 0,
        'mse': 0,
        'mae': 0
    }
    total_instance = [0 for _ in range(config['max_ball_round'])]
    total_record = {
        'ce': [0 for _ in range(config['max_ball_round'])],
        'mse': [0 for _ in range(config['max_ball_round'])],
        'mae': [0 for _ in range(config['max_ball_round'])],
    }

    with torch.no_grad():
        for item in tqdm(data_loader):
            batch_input_shot, batch_input_x, batch_input_y, batch_input_player = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device)
            batch_target_shot, batch_target_x, batch_target_y, batch_target_player = item[4].to(device), item[5].to(device), item[6].to(device), item[7].to(device)
            seq_len, seq_sets = item[8].to(device), item[9].to(device)
            
            for batch_idx in range(batch_input_shot.shape[0]):
                if seq_len[batch_idx] <= encode_length:
                    continue
                loss = {
                    'total': 1e6,
                    'entropy': 1e6,
                    'mse': 1e6,
                    'mae': 1e6
                }

                record = {
                    'ce': None,
                    'mse': None,
                    'mae': None
                }

                # encoding stage
                input_shot = batch_input_shot[batch_idx][:encode_length].unsqueeze(0)
                input_x = batch_input_x[batch_idx][:encode_length].unsqueeze(0)
                input_y = batch_input_y[batch_idx][:encode_length].unsqueeze(0)
                input_player = batch_input_player[batch_idx][:encode_length].unsqueeze(0)

                if model_type == 'Transformer' or model_type == 'Transformer_dis':
                    encoder_output = encoder(input_shot, input_x, input_y, input_player)
                elif model_type == 'ShuttleNet' or model_type == 'ours_rm_taa':
                    encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player)
                elif model_type == 'ours_p2r':
                    encode_output = encoder(input_shot, input_x, input_y, input_player)
                elif model_type == 'ours_r2p':
                    encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player)
                else:
                    raise NotImplementedError

                for sample_id in range(samples):
                    current_loss = {
                        'total': 0,
                        'entropy': 0,
                        'mse': 0,
                        'mae': 0
                    }

                    current_record = {
                        'ce': [0 for _ in range(config['max_ball_round'])],
                        'mse': [0 for _ in range(config['max_ball_round'])],
                        'mae': [0 for _ in range(config['max_ball_round'])],
                    }

                    # decoding stage
                    for seq_idx in range(encode_length, seq_len[batch_idx]):
                        target_shot = batch_target_shot[batch_idx][seq_idx]
                        target_x = batch_target_x[batch_idx][seq_idx]
                        target_y = batch_target_y[batch_idx][seq_idx]
                        target_player = batch_target_player[batch_idx][seq_idx].unsqueeze(0).unsqueeze(0)
                        if seq_idx == encode_length:
                            input_shot = batch_input_shot[batch_idx][seq_idx].unsqueeze(0).unsqueeze(0)
                            input_x = batch_input_x[batch_idx][seq_idx].unsqueeze(0).unsqueeze(0)
                            input_y = batch_input_y[batch_idx][seq_idx].unsqueeze(0).unsqueeze(0)
                            input_player = batch_input_player[batch_idx][seq_idx].unsqueeze(0).unsqueeze(0)
                            input_target_player = batch_target_player[batch_idx][seq_idx].unsqueeze(0).unsqueeze(0)
                        else:
                            # use its own predictions as the next input
                            input_shot = torch.cat((input_shot, prev_shot), dim=-1)
                            input_x = torch.cat((input_x, prev_x), dim=-1)
                            input_y = torch.cat((input_y, prev_y), dim=-1)
                            input_player = torch.cat((input_player, prev_player), dim=-1)
                            input_target_player = torch.cat((input_target_player, target_player), dim=-1)
                        
                        if model_type == 'Transformer' or model_type == 'Transformer_dis':
                            output_xy, output_shot_logits = decoder(input_shot, input_x, input_y, input_player, encoder_output, input_target_player)
                        elif model_type == 'ShuttleNet' or model_type == 'ours_rm_taa':
                            output_xy, output_shot_logits = decoder(input_shot, input_x, input_y, input_player, encode_local_output, encode_global_A, encode_global_B, input_target_player)
                        elif model_type == 'ours_p2r':
                            output_xy, output_shot_logits = decoder(input_shot, input_x, input_y, input_player, encode_output, target_player)
                        elif model_type == 'ours_r2p':
                            output_xy, output_shot_logits = decoder(input_shot, input_x, input_y, input_player, encode_global_A, encode_global_B, target_player)
                        else:
                            raise NotImplementedError

                        sx = torch.exp(output_xy[:, -1, 2]) #sx
                        sy = torch.exp(output_xy[:, -1, 3]) #sy
                        corr = torch.tanh(output_xy[:, -1, 4]) #corr
                        
                        cov = torch.zeros(2, 2).cuda(output_xy.device)
                        cov[0, 0]= sx * sx
                        cov[0, 1]= corr * sx * sy
                        cov[1, 0]= corr * sx * sy
                        cov[1, 1]= sy * sy
                        mean = output_xy[:, -1, 0:2]
                        
                        mvnormal = torchdist.MultivariateNormal(mean, cov)
                        output_xy = mvnormal.sample().unsqueeze(0)

                        # greedy
                        # _, output_shot = torch.topk(output_shot_logits, 1)

                        # sampling
                        shot_prob = F.softmax(output_shot_logits, dim=-1)
                        output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)

                        while output_shot[0, -1, 0] == 0:
                            output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)

                        gold_xy = torch.tensor([target_x, target_y]).unsqueeze(0).to(device, dtype=torch.float)

                        prev_shot = output_shot[:, -1, :]
                        prev_x = output_xy[:, -1, 0].unsqueeze(1)
                        prev_y = output_xy[:, -1, 1].unsqueeze(1)
                        prev_player = target_player.clone()

                        if sample_id == 0:
                            total_instance[seq_idx] += 1

                        current_loss['entropy'] += criterion['entropy'](output_shot_logits[:, -1, :], target_shot.unsqueeze(0))
                        current_loss['mse'] += criterion['mse'](output_xy[:, -1, :], gold_xy)
                        current_loss['mae'] += criterion['mae'](output_xy[:, -1, :], gold_xy)

                        current_record['ce'][seq_idx] += criterion['entropy'](output_shot_logits[:, -1, :], target_shot.unsqueeze(0)).item()
                        current_record['mse'][seq_idx] += criterion['mse'](output_xy[:, -1, :], gold_xy).item()
                        current_record['mae'][seq_idx] += criterion['mae'](output_xy[:, -1, :], gold_xy).item()

                    current_loss['total'] = current_loss['entropy'] + current_loss['mse'] + current_loss['mae']
                    if current_loss['total'] <= loss['total']:
                    # if current_loss['entropy'] + current_loss['mse'] <= loss['entropy'] + loss['mse']:
                        loss['total'] = current_loss['total'].clone()
                        loss['entropy'] = current_loss['entropy'].clone()
                        loss['mse'] = current_loss['mse'].clone()
                        loss['mae'] = current_loss['mae'].clone()

                        record['ce'] = current_record['ce'].copy()
                        record['mse'] = current_record['mse'].copy()
                        record['mae'] = current_record['mae'].copy()

                total_loss['total'] += loss['total'].item()
                total_loss['entropy'] += loss['entropy'].item()
                total_loss['mse'] += loss['mse'].item()
                total_loss['mae'] += loss['mae'].item()

                for key, value in total_record.items():
                    total_record[key] = [total+now for total, now in zip(total_record[key], record[key])]

    total_count = sum(total_instance)
    total_loss['count'] = total_count
    total_loss['total'] = round(total_loss['total'] / total_count, 4)
    total_loss['entropy'] = round(total_loss['entropy'] / total_count, 4)
    total_loss['mse'] = round(total_loss['mse'] / total_count, 4)
    total_loss['mae'] = round(total_loss['mae'] / total_count, 4)

    for key, value in total_record.items():
        total_record[key] = [round(total/instance, 4) if instance != 0 else 0 for total, instance in zip(total_record[key], total_instance)]
    
    return total_loss, total_record


def evaluation_rnn(data_loader, encoder, decoder, criterion, config, model_type, samples=1, device="cpu"):
    encode_length = config['encode_length']
    encoder.eval(), decoder.eval()
    total_loss = {
        'total': 0,
        'entropy': 0,
        'mse': 0,
        'mae': 0
    }
    total_instance = [0 for _ in range(config['max_ball_round'])]

    total_record = {
        'ce': [0 for _ in range(config['max_ball_round'])],
        'mse': [0 for _ in range(config['max_ball_round'])],
        'mae': [0 for _ in range(config['max_ball_round'])],
    }

    with torch.no_grad():
        # for item in tqdm(data_loader):
        for item in tqdm(data_loader):
            batch_input_shot, batch_input_x, batch_input_y, batch_input_player = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device)
            batch_target_shot, batch_target_x, batch_target_y, batch_target_player = item[4].to(device), item[5].to(device), item[6].to(device), item[7].to(device)
            seq_len, seq_sets = item[8].to(device), item[9].to(device)
            
            for batch_idx in range(batch_input_shot.shape[0]):
                if seq_len[batch_idx] <= encode_length:
                    # since encode 3, at least needs 3 + 1 decode input + 1 prediction
                    continue

                loss = {
                    'total': 1e6,
                    'entropy': 1e6,
                    'mse': 1e6,
                    'mae': 1e6
                }

                record = {
                    'ce': None,
                    'mse': None,
                    'mae': None
                }

                hidden = torch.zeros(config['n_layers'] * config['num_directions'], 1, config['encode_dim']).to(device)
                cell = torch.zeros(config['n_layers'] * config['num_directions'], 1, config['encode_dim']).to(device)
                hidden_p1, cell_p1 = None, None

                # encoding stage
                for encode_idx in range(encode_length):
                    input_shot = batch_input_shot[batch_idx][encode_idx].unsqueeze(0).unsqueeze(0)
                    input_x = batch_input_x[batch_idx][encode_idx].unsqueeze(0).unsqueeze(0)
                    input_y = batch_input_y[batch_idx][encode_idx].unsqueeze(0).unsqueeze(0)
                    input_player = batch_input_player[batch_idx][encode_idx].unsqueeze(0).unsqueeze(0)

                    if model_type == 'LSTM':
                        output, hidden, cell = encoder(input_shot, input_x, input_y, input_player, hidden, cell)
                    elif model_type == 'CFLSTM':
                        output, hidden, hidden_p1, cell, cell_p1 = encoder(input_shot, input_x, input_y, input_player, hidden, cell, hidden_p1, cell_p1, mode='inference')
                    else:
                        raise NotImplementedError

                for sample_id in range(samples):
                    if model_type == 'CFLSTM':
                        current_hidden, current_hidden_p1, current_cell, current_cell_p1 = hidden.clone(), hidden_p1.clone(), cell.clone(), cell_p1.clone()
                    elif model_type == 'LSTM':
                        current_hidden, current_cell = hidden.clone(), cell.clone()
                    else:
                        raise NotImplementedError

                    current_loss = {
                        'total': 0,
                        'entropy': 0,
                        'mse': 0,
                        'mae': 0
                    }

                    current_record = {
                        'ce': [0 for _ in range(config['max_ball_round'])],
                        'mse': [0 for _ in range(config['max_ball_round'])],
                        'mae': [0 for _ in range(config['max_ball_round'])],
                    }

                    for seq_idx in range(encode_length, seq_len[batch_idx]):
                        if seq_idx == encode_length:
                            input_shot = batch_input_shot[batch_idx][seq_idx].unsqueeze(0).unsqueeze(0)
                            input_x = batch_input_x[batch_idx][seq_idx].unsqueeze(0).unsqueeze(0)
                            input_y = batch_input_y[batch_idx][seq_idx].unsqueeze(0).unsqueeze(0)
                            input_player = batch_input_player[batch_idx][seq_idx].unsqueeze(0).unsqueeze(0)
                        else:
                            # use its own predictions as the next input
                            input_shot = prev_shot.clone()
                            input_x = prev_x.clone()
                            input_y = prev_y.clone()
                            input_player = prev_player.unsqueeze(0).unsqueeze(0).clone()

                        target_shot = batch_target_shot[batch_idx][seq_idx]
                        target_x = batch_target_x[batch_idx][seq_idx]
                        target_y = batch_target_y[batch_idx][seq_idx]
                        target_player = batch_target_player[batch_idx][seq_idx]

                        if model_type == 'LSTM':
                            output_xy, output_shot_logits, current_hidden, current_cell = decoder(input_shot, input_x, input_y, input_player, target_player, current_hidden, current_cell)
                        elif model_type == 'CFLSTM':
                            output_xy, output_shot_logits, current_hidden, current_hidden_p1, current_cell, current_cell_p1 = decoder(input_shot, input_x, input_y, input_player, target_player, current_hidden, current_cell, current_hidden_p1, current_cell_p1, mode='inference')
                        else:
                            raise NotImplementedError

                        sx = torch.exp(output_xy[:, -1, 2]) #sx
                        sy = torch.exp(output_xy[:, -1, 3]) #sy
                        corr = torch.tanh(output_xy[:, -1, 4]) #corr
                        
                        cov = torch.zeros(2, 2).cuda(output_xy.device)
                        cov[0, 0]= sx * sx
                        cov[0, 1]= corr * sx * sy
                        cov[1, 0]= corr * sx * sy
                        cov[1, 1]= sy * sy
                        mean = output_xy[:, -1, 0:2]
                        
                        mvnormal = torchdist.MultivariateNormal(mean, cov)
                        output_xy = mvnormal.sample().unsqueeze(0)
                        
                        # greedy
                        # _, output_shot = torch.topk(output_shot_logits, 1)

                        # temperature sampling
                        shot_prob = F.softmax(output_shot_logits, dim=-1)
                        output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)

                        while output_shot[0, -1, 0] == 0:
                            output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)

                        gold_xy = torch.tensor([target_x, target_y]).unsqueeze(0).to(device, dtype=torch.float)

                        prev_shot = output_shot[:, -1, :]
                        prev_x = output_xy[:, -1, 0].unsqueeze(1)
                        prev_y = output_xy[:, -1, 1].unsqueeze(1)
                        prev_player = target_player.clone()

                        if sample_id == 0:
                            total_instance[seq_idx] += 1

                        current_loss['entropy'] += criterion['entropy'](output_shot_logits[:, -1, :], target_shot.unsqueeze(0))
                        current_loss['mse'] += criterion['mse'](output_xy[:, -1, :], gold_xy)
                        current_loss['mae'] += criterion['mae'](output_xy[:, -1, :], gold_xy)

                        current_record['ce'][seq_idx] += criterion['entropy'](output_shot_logits[:, -1, :], target_shot.unsqueeze(0)).item()
                        current_record['mse'][seq_idx] += criterion['mse'](output_xy[:, -1, :], gold_xy).item()
                        current_record['mae'][seq_idx] += criterion['mae'](output_xy[:, -1, :], gold_xy).item()

                    current_loss['total'] = current_loss['entropy'] + current_loss['mse'] + current_loss['mae']
                    if current_loss['total'] <= loss['total']:
                    # if current_loss['entropy'] + current_loss['mse'] <= loss['entropy'] + loss['mse']:
                        loss['total'] = current_loss['total'].clone()
                        loss['entropy'] = current_loss['entropy'].clone()
                        loss['mse'] = current_loss['mse'].clone()
                        loss['mae'] = current_loss['mae'].clone()

                        record['ce'] = current_record['ce'].copy()
                        record['mse'] = current_record['mse'].copy()
                        record['mae'] = current_record['mae'].copy()

                total_loss['total'] += loss['total'].item()
                total_loss['entropy'] += loss['entropy'].item()
                total_loss['mse'] += loss['mse'].item()
                total_loss['mae'] += loss['mae'].item()

                for key, value in total_record.items():
                    total_record[key] = [total+now for total, now in zip(total_record[key], record[key])]

    total_count = sum(total_instance)
    total_loss['count'] = total_count
    total_loss['total'] = round(total_loss['total'] / total_count, 4)
    total_loss['entropy'] = round(total_loss['entropy'] / total_count, 4)
    total_loss['mse'] = round(total_loss['mse'] / total_count, 4)
    total_loss['mae'] = round(total_loss['mae'] / total_count, 4)

    for key, value in total_record.items():
        total_record[key] = [round(total/instance, 4) if instance != 0 else 0 for total, instance in zip(total_record[key], total_instance)]
    
    return total_loss, total_record


def save(encoder, decoder, config, k_fold_index, epoch=None):
    output_folder_name = config['output_folder_name'] + str(k_fold_index) + '/'
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    if epoch is None:
        encoder_name = output_folder_name + 'encoder'
        decoder_name = output_folder_name + 'decoder'
        config_name = output_folder_name + 'config'
    else:
        encoder_name = output_folder_name + str(epoch) + 'encoder'
        decoder_name = output_folder_name + str(epoch) + 'decoder'
        config_name = output_folder_name + str(epoch) + 'config'
    
    torch.save(encoder.state_dict(), encoder_name)
    torch.save(decoder.state_dict(), decoder_name)
    with open(config_name, 'w') as config_file:
        config_file.write(str(config))