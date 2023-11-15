import os
import torch
import torch.nn.functional as F
import torch.distributions.multivariate_normal as torchdist
from tqdm import tqdm
import numpy as np
from copy import deepcopy

### GMM ###
import torch.nn.functional as F
import torch.distributions as D
GMM_NUM = 5
####


def evaluation(data_loader, encoder, decoder, criterion, config, samples=1, device="cpu"):
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

                encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player)


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
                        
                        output_xy, output_shot_logits = decoder(input_shot, input_x, input_y, input_player, encode_local_output, encode_global_A, encode_global_B, input_target_player)


                        #sx = torch.exp(output_xy[:, -1, 2]) #sx
                        #sy = torch.exp(output_xy[:, -1, 3]) #sy
                        #corr = torch.tanh(output_xy[:, -1, 4]) #corr
                        #
                        #cov = torch.zeros(2, 2).cuda(output_xy.device)
                        #cov[0, 0]= sx * sx
                        #cov[0, 1]= corr * sx * sy
                        #cov[1, 0]= corr * sx * sy
                        #cov[1, 1]= sy * sy
                        #mean = output_xy[:, -1, 0:2]
                        #
                        #mvnormal = torchdist.MultivariateNormal(mean, cov)
                        #output_xy = mvnormal.sample().unsqueeze(0)
                        # using GMM
                        epsilon = 1e-20
                        output_xy[output_xy==0] = epsilon
                        output_xy = output_xy[:,-1,:]

                        """
                        land GMM loss
                        """
                        land_gmm = D.MixtureSameFamily(D.Categorical(F.softmax(output_xy[:,:GMM_NUM], dim=-1)),\
                        D.Independent(D.Normal(torch.reshape(output_xy[:,GMM_NUM:GMM_NUM*3],(-1,GMM_NUM,2)),\
                        torch.reshape(torch.abs(torch.clamp(output_xy[:,GMM_NUM*3:],-0.1,0.1)),(-1,GMM_NUM,2))), 1))
                        output_xy = land_gmm.sample((1,))
                        # greedy
                        # _, output_shot = torch.topk(output_shot_logits, 1)

                        # sampling
                        shot_prob = F.softmax(output_shot_logits, dim=-1)
                        output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)

                        while output_shot[0, -1, 0] == 0:
                            output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)

                        gold_xy = torch.tensor([target_x, target_y]).unsqueeze(0).to(device, dtype=torch.float)

                        shotTypeName = config['uniques_type'][output_shot[0,0,-1]-1]

                        #x = output_xy[0,-1,0]
                        #y = output_xy[0,-1,1]
                        #mean_x, std_x = 175., 82.
                        #mean_y, std_y = 467., 192.
                        #x = x * std_x + mean_x
                        #y = y * std_y + mean_y

                        #print(f"predict (x, y): ({x: .2f}, {y: .2f}), shot type: {shotTypeName}")
                        #input()

                        prev_shot = output_shot[:, -1, :]
                        prev_x = output_xy[:, -1, 0].unsqueeze(1)
                        prev_y = output_xy[:, -1, 1].unsqueeze(1)
                        prev_player = target_player.clone()

                        if sample_id == 0:
                            total_instance[seq_idx] += 1

                        # conver to long to avoid RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Int'
                        target_shot = target_shot.type(torch.LongTensor).to(device)
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

def predict(data_loader, encoder, decoder, config, samples=1, device="cpu"):
    print(config['encode_length'])
    encode_length = config['encode_length']
    encoder.eval(), decoder.eval()

    shot_type, landing_x, landing_y, player = data_loader

    #split samples to multiple batches
    batches = []
    maxBatchSize = config['max_ball_round'] - encode_length
    while samples >= maxBatchSize:
        batches.append(maxBatchSize)
        samples -= maxBatchSize
    if samples != 0:
        batches.append(samples)

    shot_type = shot_type[:5]
    landing_x = landing_x[:5]
    landing_y = landing_y[:5]
    player = player[:5]
    landing_gmm = np.zeros(5, dtype=object)
    #landing_sx = np.full(5,0.00001)
    #landing_sy = np.full(5,0.00001)
    #landing_tho = np.zeros(5)
    shot_distribution = [[], [], [], [], []]

    with torch.no_grad():
        for batch in batches:
            # encoding stage
            input_shot = torch.tensor(shot_type)[-encode_length-1:].to(device).view(1,-1)
            input_x = torch.tensor(landing_x)[-encode_length-1:].to(device).view(1,-1)
            input_y = torch.tensor(landing_y)[-encode_length-1:].to(device).view(1,-1)
            input_player = torch.tensor(player)[-encode_length-1:].to(device).view(1,-1)
            #print(input_shot, input_x, input_y, input_player)

            encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player)

            shot_type_predict = np.zeros(batch, dtype = int)
            landing_gmm_predict = np.zeros(batch, dtype = object)
            landing_x_predict = np.zeros(batch, dtype = np.float32)
            landing_y_predict = np.zeros(batch, dtype = np.float32)
            #landing_sx_predict = np.zeros(batch, dtype = np.float32)
            #landing_sy_predict = np.zeros(batch, dtype = np.float32)
            #landing_tho_predict = np.zeros(batch, dtype = np.float32)
            player_predict = np.ones(batch, dtype = int)
            shot_distribution_predict = []
            for i in range(player[-1] == 2,len(player_predict),2):
                player_predict[i] = 2

            for i in range(batch):
                if i == 0:
                    input_shot = torch.tensor(shot_type)[-1:].to(device).view(1,-1)
                    input_x = torch.tensor(landing_x)[-1:].to(device).view(1,-1)
                    input_y = torch.tensor(landing_y)[-1:].to(device).view(1,-1)
                    input_player = torch.tensor(player)[-1:].to(device).view(1,-1)
                    input_target_player = torch.tensor(1 if input_player[0][-1] == 2 else 2).to(device).view(1,1)
                else:
                    input_shot = torch.cat((input_shot, last_shot), dim=-1)
                    input_x = torch.cat((input_x, last_x), dim=-1)
                    input_y = torch.cat((input_y, last_y), dim=-1)
                    input_player = torch.cat((input_player, last_player), dim=-1)
                    target_player = torch.tensor(1 if input_player[0][-1] == 2 else 2).to(device).view(1,1)
                    input_target_player = torch.cat((input_target_player, target_player), dim=-1)
                output_xy, output_shot_logits = decoder(input_shot, input_x, input_y, input_player, encode_local_output, encode_global_A, encode_global_B, input_target_player)
                
                #output_xy = [x, y, Sx, Sy, ρ]
                #print(output_xy, output_shot_logits)

                epsilon = 1e-20
                output_xy[output_xy==0] = epsilon
                output_xy = output_xy[:,-1,:]

                """
                land GMM loss
                """
                weights = F.softmax(output_xy[:,:GMM_NUM], dim=-1)
                means = torch.reshape(output_xy[:,GMM_NUM:GMM_NUM*3],(-1,GMM_NUM,2))
                stdevs = torch.reshape(torch.abs(torch.clamp(output_xy[:,GMM_NUM*3:],-0.1,0.1)),(-1,GMM_NUM,2))
                gmm_param = (weights.tolist(), means.tolist(), stdevs.tolist())
                land_gmm = D.MixtureSameFamily(D.Categorical(weights),D.Independent(D.Normal(means, stdevs), 1))
                x, y = land_gmm.sample((1,)).unsqueeze(0)

                #sx = torch.exp(output_xy[:, -1, 2]) #sx
                #sy = torch.exp(output_xy[:, -1, 3]) #sy
                #corr = torch.tanh(output_xy[:, -1, 4]) #corr

                #cov = torch.zeros(2, 2).cuda(output_xy.device)
                #cov[0, 0]= sx * sx
                #cov[0, 1]= corr * sx * sy
                #cov[1, 0]= corr * sx * sy
                #cov[1, 1]= sy * sy
                #mean = output_xy[:, -1, 0:2]

                #mvnormal = torchdist.MultivariateNormal(mean, cov)
                #output_xy = mvnormal.sample().unsqueeze(0)

                # sampling
                shot_prob = F.softmax(output_shot_logits, dim=-1)
                distribution = shot_prob[0,0].cpu().detach().numpy().tolist()[1:]

                while True:
                    output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)
                    if output_shot[0, -1, 0] != 0: # 0 is padding
                        break

                output_shot = output_shot[0, -1, 0]

                last_x = output_xy[0, -1, 0].view(1,-1)
                last_y = output_xy[0, -1, 1].view(1,-1)
                last_shot = output_shot.view(1,-1)
                last_player = input_player[0,-1].view(1,-1)

                type = output_shot.item()-1

                #x = output_xy[0,-1,0]
                #y = output_xy[0,-1,1]

                #predictShot.append((x.item(),y.item(),shotType))
                shot_type_predict[i] = type
                landing_gmm_predict[i] = deepcopy(gmm_param)
                landing_x_predict[i] = x.item()
                landing_y_predict[i] = y.item()
                #landing_sx_predict[i] = sx.item()
                #landing_sy_predict[i] = sy.item()
                #landing_tho_predict[i] = corr.item()

                shot_distribution_predict.append(distribution)
            shot_type = np.append(shot_type,shot_type_predict)
            landing_gmm = np.append(landing_gmm, landing_gmm_predict)
            landing_x = np.append(landing_x,landing_x_predict)
            landing_y = np.append(landing_y,landing_y_predict)
            player = np.append(player,player_predict)
            #landing_sx = np.append(landing_sx, landing_sx_predict)
            #landing_sy = np.append(landing_sy, landing_sy_predict)
            #landing_tho = np.append(landing_tho, landing_tho_predict)
            shot_distribution += shot_distribution_predict
            

    for i in range(4):
        shot_type[i] -= 1

    output = []
    for x, y, shot, p, gmm, shot_prob in zip(landing_x, landing_y, shot_type, player, landing_gmm, shot_distribution):
        shotTypeName = config['uniques_type'][shot]
        output.append((x, y, shotTypeName, p, gmm, shot_prob))
        
        #mean_x, std_x = 175., 82.
        #mean_y, std_y = 467., 192.
        #x = x * std_x + mean_x
        #y = y * std_y + mean_y
        #print(f"predict (x, y): ({x: .2f}, {y: .2f}), shot type: {shotTypeName}")

    return output

def predictAgent(data_loader, encoder, decoder, config, device="cpu"):
    shot_type, landing_x, landing_y, player = data_loader
    #print(config['encode_length'])
    GMM_NUM = config['num_GMM']
    encode_length = config['encode_length']
    max_length = config['max_ball_round']
    encoder.eval(), decoder.eval()

    shot_type = shot_type[:max_length-2]
    landing_x = landing_x[:max_length-2]
    landing_y = landing_y[:max_length-2]
    player = player[:max_length-2]

    with torch.no_grad():
        # encoding stage
        input_shot = torch.tensor(shot_type)[:encode_length].to(device).view(1,-1)
        input_x = torch.tensor(landing_x)[:encode_length].to(device).view(1,-1)
        input_y = torch.tensor(landing_y)[:encode_length].to(device).view(1,-1)
        input_player = torch.tensor(player)[:encode_length].to(device).view(1,-1)
        #print(input_shot, input_x, input_y, input_player)

        encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player)

        input_shot = torch.tensor(shot_type)[encode_length-1:].to(device).view(1,-1)
        input_x = torch.tensor(landing_x)[encode_length-1:].to(device).view(1,-1)
        input_y = torch.tensor(landing_y)[encode_length-1:].to(device).view(1,-1)
        input_player = torch.tensor(player)[encode_length-1:].to(device).view(1,-1)
        input_target_player = torch.tensor(1 if input_player[0][-1] == 2 else 2).to(device).view(1,1)

        output_xy, output_shot_logits = decoder(input_shot, input_x, input_y, input_player, encode_local_output, encode_global_A, encode_global_B, input_target_player)
        
        epsilon = 1e-20
        output_xy[output_xy==0] = epsilon
        output_xy = output_xy[:,-1,:]

        """
        land GMM loss
        """
        weights = F.softmax(output_xy[:,:GMM_NUM], dim=-1)
        means = torch.reshape(output_xy[:,GMM_NUM:GMM_NUM*3],(-1,GMM_NUM,2))
        stdevs = torch.reshape(torch.abs(torch.clamp(output_xy[:,GMM_NUM*3:],-0.1,0.1)),(-1,GMM_NUM,2))
        gmm_param = (weights.tolist(), means.tolist(), stdevs.tolist())
        land_gmm = D.MixtureSameFamily(D.Categorical(weights),D.Independent(D.Normal(means, stdevs), 1))

        #output_xy = [x, y, Sx, Sy, ρ]
        #print(output_xy, output_shot_logits)

        #sx = torch.exp(output_xy[:, -1, 2]) #sx
        #sy = torch.exp(output_xy[:, -1, 3]) #sy
        #corr = torch.tanh(output_xy[:, -1, 4]) #corr

        #cov = torch.zeros(2, 2).cuda(output_xy.device)
        #cov[0, 0]= sx * sx
        #cov[0, 1]= corr * sx * sy
        #cov[1, 0]= corr * sx * sy
        #cov[1, 1]= sy * sy
        #mean = output_xy[:, -1, 0:2]

        #mvnormal = torchdist.MultivariateNormal(mean, cov)
        #output_xy = mvnormal.sample().unsqueeze(0)

        # sampling
        shot_prob = F.softmax(output_shot_logits, dim=-1)
        shot_distribution = shot_prob[0,0].cpu().detach().numpy().tolist()[1:]

        while True:
            output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)
            if output_shot[0, -1, 0] != 0: # 0 is padding
                break

        output_shot = output_shot[0, -1, 0]

        #last_x = output_xy[0, -1, 0].view(1,-1)
        #last_y = output_xy[0, -1, 1].view(1,-1)
        #last_shot = output_shot.view(1,-1)
        #last_player = input_player[0,-1].view(1,-1)

        type = output_shot.item()-1

        #x = output_xy[0,-1,0]
        #y = output_xy[0,-1,1]
        x, y = land_gmm.sample((1,)).squeeze(0)[0]

        # output = (type, x.item(), y.item(), sx.item(), sy.item(), corr.item())
        #predictShot.append((x.item(),y.item(),shotType))
    return int(type), x.item(), y.item(), shot_distribution, gmm_param


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
    with open(config_name, 'w', encoding='utf8') as config_file:
        config_file.write(str(config))