import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributions.multivariate_normal as torchdist
PAD = 0

def Gaussian2D_loss(V_pred, V_trgt):
    #mux, muy, sx, sy, corr
    #assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:, 0] - V_pred[:, 0]
    normy = V_trgt[:, 1] - V_pred[:, 1]

    sx = torch.exp(V_pred[:, 2]) #sx
    sy = torch.exp(V_pred[:, 3]) #sy
    corr = torch.tanh(V_pred[:, 4]) #corr
    
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.sum(result)
    
    return result

def train(train_dataloader, valid_dataloader, encoder, decoder, location_criterion, shot_type_criterion, encoder_optimizer, decoder_optimizer, args, device="cpu"):
    encode_length = args['encode_length']    
    best_loss = 1e6
    best_loss_location = 1e6
    best_loss_type = 1e6
    for epoch in tqdm(range(args['epochs'])):
        encoder.train(), decoder.train()
        for rally, target in train_dataloader:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # rally information
            player = rally[0].to(device).long()
            shot_type = rally[1].to(device).long()
            player_A_x = rally[2].to(device)
            player_A_y = rally[3].to(device)
            player_B_x = rally[4].to(device)
            player_B_y  = rally[5].to(device)
            
            # target information
            target_A_x = target[0].to(device)
            target_A_y = target[1].to(device)
            target_B_x = target[2].to(device)
            target_B_y = target[3].to(device)
            target_type = target[4].to(device)
            
            encoder_player = player[:, 0:2]
            encoder_shot_type = shot_type[:, :encode_length-1]
            encoder_player_A_x = player_A_x[:, :encode_length]
            encoder_player_A_y = player_A_y[:, :encode_length]
            encoder_player_B_x = player_B_x[:, :encode_length]
            encoder_player_B_y = player_B_y[:, :encode_length]

            encode_node_embedding, adjacency_matrix = encoder(encoder_player, encoder_shot_type, encoder_player_A_x, encoder_player_A_y, encoder_player_B_x, encoder_player_B_y, encode_length)
            
            all_A_predictions = []
            all_B_predictions = []
            all_shot_type_predictions = []
            decode_node_embedding = encode_node_embedding.clone()

            decoder_player = player[:, 0:2]

            first = True
            for step in range(encode_length, args['max_length']+1):
                decoder_shot_type = shot_type[:, step-1:step]
                decoder_player_A_x = player_A_x[:, step-1:step]
                decoder_player_A_y = player_A_y[:, step-1:step]
                decoder_player_B_x = player_B_x[:, step-1:step]
                decoder_player_B_y = player_B_y[:, step-1:step]

                predict_xy, predict_shot_type_logit, adjacency_matrix, decode_node_embedding = decoder(decoder_player, step+1, decode_node_embedding, adjacency_matrix, 
                                                                                                    decoder_player_A_x, decoder_player_A_y, decoder_player_B_x, decoder_player_B_y, 
                                                                                                    shot_type=decoder_shot_type, train=True, first=first)
                all_A_predictions.append(predict_xy[:, 0, :])
                all_B_predictions.append(predict_xy[:, 1, :])
                all_shot_type_predictions.append(predict_shot_type_logit)
                first = False
            
            predict_A_xy = torch.stack(all_A_predictions, dim=1)
            predict_B_xy = torch.stack(all_B_predictions, dim=1)            
            predict_shot_type_logit = torch.stack(all_shot_type_predictions, dim=1)
            
            target_A_x = target_A_x[:, encode_length-1:]
            target_A_y = target_A_y[:, encode_length-1:]
            target_B_x = target_B_x[:, encode_length-1:]
            target_B_y = target_B_y[:, encode_length-1:]
            target_type = target_type[:, encode_length-2:-1]

            pad_mask = (target_type!=PAD)
            predict_A_xy  = predict_A_xy[pad_mask]
            predict_B_xy  = predict_B_xy[pad_mask]
            predict_shot_type_logit = predict_shot_type_logit[pad_mask]

            target_A_x = target_A_x[pad_mask]
            target_A_y = target_A_y[pad_mask]
            target_B_x = target_B_x[pad_mask]
            target_B_y = target_B_y[pad_mask]
            target_type = target_type[pad_mask]

            gold_A_xy = torch.cat((target_A_x.unsqueeze(-1), target_A_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
            gold_B_xy = torch.cat((target_B_x.unsqueeze(-1), target_B_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
                       
            loss_type = shot_type_criterion(predict_shot_type_logit, target_type)
            loss_location = (Gaussian2D_loss(predict_A_xy, gold_A_xy) + Gaussian2D_loss(predict_B_xy, gold_B_xy)) / 2
            loss = loss_location + loss_type
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

        # evaluate_loss, loss_location, loss_type = evaluate(valid_dataloader, encoder, decoder, location_criterion, shot_type_criterion, args, device=device)
        if loss < best_loss:
            best_loss = loss
            best_loss_location = loss_location
            best_loss_type = loss_type
        # if evaluate_loss < best_loss:
        #     best_loss = evaluate_loss
        #     best_loss_location = loss_location
        #     best_loss_type = loss_type
        #     save(encoder, decoder, args)
    save(encoder, decoder, args)
    return best_loss, best_loss_location, best_loss_type


def evaluate(test_dataloader, encoder, decoder, location_MSE_criterion, location_MAE_criterion, shot_type_criterion, args, device="cpu"):
    encode_length = args['encode_length']
    encoder.eval(), decoder.eval()

    total_instance = 0
    total_loss = 0
    total_loss_MSE_location = 0
    total_loss_MAE_location = 0
    total_loss_type = 0

    with torch.no_grad():
        for rally, target in tqdm(test_dataloader):
            best_loss = 1e9
            best_location_MSE_loss = 0
            best_location_MAE_loss = 0
            best_type_loss = 0             
            for sample_index in range(args['sample_num']):
                tmp_rally_location_MSE_loss = 0
                tmp_rally_location_MAE_loss = 0
                tmp_rally_type_loss = 0

                # rally information
                player = rally[0].to(device).long()
                shot_type = rally[1].to(device).long()
                player_A_x = rally[2].to(device)
                player_A_y = rally[3].to(device)
                player_B_x = rally[4].to(device)
                player_B_y  = rally[5].to(device)
                length = rally[6]

                # target information
                target_A_x = target[0].to(device)
                target_A_y = target[1].to(device)
                target_B_x = target[2].to(device)
                target_B_y = target[3].to(device)
                target_type = target[4].to(device)

                encoder_player = player[:, 0:2]
                encoder_shot_type = shot_type[:, :encode_length-1]
                encoder_player_A_x = player_A_x[:, :encode_length]
                encoder_player_A_y = player_A_y[:, :encode_length]
                encoder_player_B_x = player_B_x[:, :encode_length]
                encoder_player_B_y = player_B_y[:, :encode_length]

                encode_node_embedding, adjacency_matrix = encoder(encoder_player, encoder_shot_type, encoder_player_A_x, encoder_player_A_y, encoder_player_B_x, encoder_player_B_y, encode_length)
                
                decoder_node_embedding = encode_node_embedding.clone()
                            
                decoder_player = player[:, 0:2]
                        
                decoder_player_A_x = player_A_x[:, encode_length-1:encode_length]
                decoder_player_A_y = player_A_y[:, encode_length-1:encode_length]
                decoder_player_B_x = player_B_x[:, encode_length-1:encode_length]
                decoder_player_B_y = player_B_y[:, encode_length-1:encode_length]

                first = True
                for sequence_index in range(encode_length, length[0]+1):
                    predict_xy, predict_shot_type_logit, adjacency_matrix, decoder_node_embedding = decoder(decoder_player, sequence_index+1, decoder_node_embedding, adjacency_matrix, 
                                                                                                            decoder_player_A_x, decoder_player_A_y, decoder_player_B_x, decoder_player_B_y,
                                                                                                            shot_type=None, train=False, first=first)

                    predict_A_xy = predict_xy[:, 0:1, :]
                    predict_B_xy = predict_xy[:, 1:2, :]
                    
                    sx = torch.exp(predict_A_xy[:, -1, 2]) #sx
                    sy = torch.exp(predict_A_xy[:, -1, 3]) #sy
                    corr = torch.tanh(predict_A_xy[:, -1, 4]) #corr                

                    cov = torch.zeros(2, 2).to(player.device)
                    cov[0, 0]= sx * sx
                    cov[0, 1]= corr * sx * sy
                    cov[1, 0]= corr * sx * sy
                    cov[1, 1]= sy * sy
                    mean = predict_A_xy[:, -1, 0:2]
                    
                    mvnormal = torchdist.MultivariateNormal(mean, cov)
                    predict_A_xy = mvnormal.sample().unsqueeze(0)

                    sx = torch.exp(predict_B_xy[:, -1, 2]) #sx
                    sy = torch.exp(predict_B_xy[:, -1, 3]) #sy
                    corr = torch.tanh(predict_B_xy[:, -1, 4]) #corr                

                    cov = torch.zeros(2, 2).to(player.device)
                    cov[0, 0]= sx * sx
                    cov[0, 1]= corr * sx * sy
                    cov[1, 0]= corr * sx * sy
                    cov[1, 1]= sy * sy
                    mean = predict_B_xy[:, -1, 0:2]
                    
                    mvnormal = torchdist.MultivariateNormal(mean, cov)
                    predict_B_xy = mvnormal.sample().unsqueeze(0)                

                    decoder_target_A_x = target_A_x[:, sequence_index-1:sequence_index]
                    decoder_target_A_y = target_A_y[:, sequence_index-1:sequence_index]
                    decoder_target_B_x = target_B_x[:, sequence_index-1:sequence_index]
                    decoder_target_B_y = target_B_y[:, sequence_index-1:sequence_index]
                    decoder_target_type = target_type[:, sequence_index-2]

                    gold_A_xy = torch.cat((decoder_target_A_x.unsqueeze(-1), decoder_target_A_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
                    gold_B_xy = torch.cat((decoder_target_B_x.unsqueeze(-1), decoder_target_B_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
                    
                    loss_MSE_A = location_MSE_criterion(predict_A_xy, gold_A_xy)
                    loss_MSE_B = location_MSE_criterion(predict_B_xy, gold_B_xy)
                    loss_MAE_A = location_MAE_criterion(predict_A_xy, gold_A_xy)
                    loss_MAE_B = location_MAE_criterion(predict_B_xy, gold_B_xy)
                    
                    loss_MSE_location = loss_MSE_A + loss_MSE_B
                    loss_MAE_location = loss_MAE_A + loss_MAE_B
                    loss_type = shot_type_criterion(predict_shot_type_logit, decoder_target_type)
                    if sample_index == 0:
                        total_instance += 1
                    tmp_rally_location_MSE_loss += loss_MSE_location.item()
                    tmp_rally_location_MAE_loss += loss_MAE_location.item()
                    tmp_rally_type_loss += loss_type.item()

                    decoder_player_A_x = predict_A_xy[:, 0, 0:1]
                    decoder_player_A_y = predict_A_xy[:, 0, 1:2]
                    decoder_player_B_x = predict_B_xy[:, 0, 0:1]
                    decoder_player_B_y = predict_B_xy[:, 0, 1:2]

                    first = False
                    
                if (tmp_rally_location_MSE_loss + tmp_rally_location_MAE_loss + tmp_rally_type_loss) < best_loss:
                    best_location_MSE_loss = tmp_rally_location_MSE_loss
                    best_location_MAE_loss = tmp_rally_location_MAE_loss
                    best_type_loss = tmp_rally_type_loss
                    best_loss = tmp_rally_location_MSE_loss + tmp_rally_location_MAE_loss + tmp_rally_type_loss
                    
            total_loss_MSE_location += best_location_MSE_loss
            total_loss_MAE_location += best_location_MAE_loss
            total_loss_type += best_type_loss
            total_loss += best_loss

    total_loss = round(total_loss / total_instance, 4)
    total_loss_type = round(total_loss_type / total_instance, 4)
    total_loss_MSE_location = round(total_loss_MSE_location / total_instance, 4)
    total_loss_MAE_location = round(total_loss_MAE_location / total_instance, 4)

    return total_loss, total_loss_MSE_location, total_loss_MAE_location, total_loss_type

def save(encoder, decoder, args):
    output_folder_name = args['model_folder']
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)
    
    torch.save(encoder.state_dict(), output_folder_name + '/encoder')
    torch.save(decoder.state_dict(), output_folder_name + '/decoder')
