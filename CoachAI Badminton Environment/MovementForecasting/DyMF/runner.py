import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributions.multivariate_normal as torchdist
from scipy.stats import multivariate_normal

### GMM ###
import torch.nn.functional as F
import torch.distributions as D
######

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
    GMM_NUM = args['num_GMM']
    best_loss = 1e6
    best_loss_location = 1e6
    best_loss_type = 1e6
    record_loss = {
        'total': [],
        'shot': [],
        'area': []
    }
    for epoch in tqdm(range(args['epochs'])):
        total_loss, total_shot_loss, total_area_loss = 0, 0, 0
        total_instance = 0
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

            encode_node_embedding, original_embedding, adjacency_matrix = encoder(encoder_player, encoder_shot_type, encoder_player_A_x, encoder_player_A_y, encoder_player_B_x, encoder_player_B_y, encode_length)

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
                
                predict_xy, predict_shot_type_logit, adjacency_matrix, decode_node_embedding, original_embedding = decoder(decoder_player, step+1, decode_node_embedding, original_embedding, adjacency_matrix, 
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

            total_instance += len(target_type)

            gold_A_xy = torch.cat((target_A_x.unsqueeze(-1), target_A_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
            gold_B_xy = torch.cat((target_B_x.unsqueeze(-1), target_B_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)

            #### GMM ####
            epsilon = 1e-20
            predict_A_xy[predict_A_xy==0] = epsilon
            predict_B_xy[predict_B_xy==0] = epsilon

            """
            land GMM loss
            """
            predictA_land_gmm = D.MixtureSameFamily(D.Categorical(F.softmax(predict_A_xy[:,:GMM_NUM], dim=-1)),\
            D.Independent(D.Normal(torch.reshape(predict_A_xy[:,GMM_NUM:GMM_NUM*3],(-1,GMM_NUM,2)),\
            torch.reshape(torch.abs(predict_A_xy[:,GMM_NUM*3:]),(-1,GMM_NUM,2))), 1))
            loss_area_A = -predictA_land_gmm.log_prob(gold_A_xy).sum()

            predictB_land_gmm = D.MixtureSameFamily(D.Categorical(F.softmax(predict_B_xy[:,:GMM_NUM], dim=-1)),\
            D.Independent(D.Normal(torch.reshape(predict_B_xy[:,GMM_NUM:GMM_NUM*3],(-1,GMM_NUM,2)),\
            torch.reshape(torch.abs(predict_B_xy[:,GMM_NUM*3:]),(-1,GMM_NUM,2))), 1))
            loss_area_B = -predictB_land_gmm.log_prob(gold_B_xy).sum()
            ########
                       
            loss_type = shot_type_criterion(predict_shot_type_logit, target_type)
            #loss_location = (Gaussian2D_loss(predict_A_xy, gold_A_xy) + Gaussian2D_loss(predict_B_xy, gold_B_xy)) / 2
            loss_location = (loss_area_A + loss_area_B) / 2
            loss = loss_location + loss_type
            loss.backward()            
            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()
            total_shot_loss += loss_type.item()
            total_area_loss += loss_location.item()

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

        total_loss = round(total_loss / total_instance, 4)
        total_shot_loss = round(total_shot_loss / total_instance, 4)
        total_area_loss = round(total_area_loss / total_instance, 4)

        record_loss['total'].append(total_loss)
        record_loss['shot'].append(total_shot_loss)
        record_loss['area'].append(total_area_loss)
    save(encoder, decoder, args)
    return best_loss, best_loss_location, best_loss_type, record_loss


def evaluate(test_dataloader, encoder, decoder, location_MSE_criterion, location_MAE_criterion, shot_type_criterion, args, device="cpu"):
    encode_length = args['encode_length']
    GMM_NUM = args['num_GMM']
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

                encode_node_embedding, original_embedding, adjacency_matrix = encoder(encoder_player, encoder_shot_type, encoder_player_A_x, encoder_player_A_y, encoder_player_B_x, encoder_player_B_y, encode_length)
                    
                decoder_node_embedding = encode_node_embedding.clone()
                            
                decoder_player = player[:, 0:2]

                decoder_player_A_x = player_A_x[:, encode_length-1:encode_length]
                decoder_player_A_y = player_A_y[:, encode_length-1:encode_length]
                decoder_player_B_x = player_B_x[:, encode_length-1:encode_length]
                decoder_player_B_y = player_B_y[:, encode_length-1:encode_length]
                
                first = True
                for sequence_index in range(encode_length, length[0]+1):
                    predict_xy, predict_shot_type_logit, adjacency_matrix, decoder_node_embedding, original_embedding = decoder(decoder_player, sequence_index+1, decoder_node_embedding, original_embedding, adjacency_matrix, 
                                                                                                                            decoder_player_A_x, decoder_player_A_y, decoder_player_B_x, decoder_player_B_y,
                                                                                                                            shot_type=None, train=False, first=first)

                    predict_A_xy = predict_xy[:, 0, :]
                    predict_B_xy = predict_xy[:, 1, :]

                    #### GMM ####
                    epsilon = 1e-20
                    predict_A_xy[predict_A_xy==0] = epsilon
                    predict_B_xy[predict_B_xy==0] = epsilon

                    """
                    land GMM loss
                    """
                    predictA_land_gmm = D.MixtureSameFamily(D.Categorical(F.softmax(predict_A_xy[:,:GMM_NUM], dim=-1)),\
                    D.Independent(D.Normal(torch.reshape(predict_A_xy[:,GMM_NUM:GMM_NUM*3],(-1,GMM_NUM,2)),\
                    torch.reshape(torch.abs(torch.clamp(predict_A_xy[:,GMM_NUM*3:],-0.3,0.3)),(-1,GMM_NUM,2))), 1))
                    predict_A_xy = predictA_land_gmm.sample((1,))

                    predictB_land_gmm = D.MixtureSameFamily(D.Categorical(F.softmax(predict_B_xy[:,:GMM_NUM], dim=-1)),\
                    D.Independent(D.Normal(torch.reshape(predict_B_xy[:,GMM_NUM:GMM_NUM*3],(-1,GMM_NUM,2)),\
                    torch.reshape(torch.abs(torch.clamp(predict_B_xy[:,GMM_NUM*3:],-0.3,0.3)),(-1,GMM_NUM,2))), 1))
                    predict_B_xy = predictB_land_gmm.sample((1,))
                    ########              

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

def predict(data, encoder, decoder, args, samples = 35, device="cpu"):
    encode_length = args['encode_length']
    GMM_NUM = args['num_GMM']
    encoder.eval(), decoder.eval()

    #split samples to multiple batches
    batches = []
    maxBatchSize = args['max_length'] - encode_length
    while samples >= maxBatchSize:
        batches.append(maxBatchSize)
        samples -= maxBatchSize
    if samples != 0:
        batches.append(samples)

    player, shot_type, player_A_x, player_A_y, player_B_x, player_B_y = data

    player = player[:5]
    shot_type = shot_type[:5]
    player_A_x = player_A_x[:5]
    player_A_y = player_A_y[:5]
    player_B_x = player_B_x[:5]
    player_A_y = player_A_y[:5]

    #all_player_A_x_record = []
    #all_player_A_y_record = []
    #all_player_B_x_record = []
    #all_player_B_y_record = []
    #all_shot_type_record = []

    all_mean_A = []
    all_mean_B = []
    all_cov_A = []
    all_cov_B = []
    
    mean_x, std_x = 175., 82.
    mean_y, std_y = 467., 192.

    with torch.no_grad():
        #player_A_x_record = []
        #player_A_y_record = []
        #player_B_x_record = []
        #player_B_y_record = []
        #shot_type_record = []

        # rally information
        #player = torch.tensor([data[0]]).to(device).long()
        #shot_type = torch.tensor([data[1]]).to(device).long()
        #player_A_x = torch.tensor([data[2]]).to(device)
        #player_A_y = torch.tensor([data[3]]).to(device)
        #player_B_x = torch.tensor([data[4]]).to(device)
        #player_B_y  = torch.tensor([data[5]]).to(device)


        #for i in range(encoder_player_A_x.size(1)):
        #    player_A_x_record.append(encoder_player_A_x[0][i].item() * std_x + mean_x)
        #    player_A_y_record.append(960 - (encoder_player_A_y[0][i].item() * std_y + mean_y))
        #    player_B_x_record.append(encoder_player_B_x[0][i].item() * std_x + mean_x)
        #    player_B_y_record.append(960 - (encoder_player_B_y[0][i].item() * std_y + mean_y))
        #for i in range(encoder_shot_type.size(1)):
        #    shot_type_record.append(encoder_shot_type[0][i].item())

        for batch in batches:
        #for rally in all_dataloader:
            
            encoder_player = torch.tensor(player)[-2:].to(device).view(1,-1)
            encoder_shot_type = torch.tensor(shot_type)[-encode_length+1:].to(device).view(1,-1)
            encoder_player_A_x = torch.tensor(player_A_x)[-encode_length:].to(device).view(1,-1)
            encoder_player_A_y = torch.tensor(player_A_y)[-encode_length:].to(device).view(1,-1)
            encoder_player_B_x = torch.tensor(player_B_x)[-encode_length:].to(device).view(1,-1)
            encoder_player_B_y = torch.tensor(player_B_y)[-encode_length:].to(device).view(1,-1)

            encode_node_embedding, original_embedding, adjacency_matrix = encoder(encoder_player, encoder_shot_type, encoder_player_A_x, encoder_player_A_y, encoder_player_B_x, encoder_player_B_y, encode_length)
                
            decoder_node_embedding = encode_node_embedding.clone()
                        
            decoder_player = torch.tensor(player)[0:2].to(device).view(1,-1)

            decoder_player_A_x = torch.tensor(player_A_x)[encode_length-1:encode_length].to(device).view(1,-1)
            decoder_player_A_y = torch.tensor(player_A_y)[encode_length-1:encode_length].to(device).view(1,-1)
            decoder_player_B_x = torch.tensor(player_B_x)[encode_length-1:encode_length].to(device).view(1,-1)
            decoder_player_B_y = torch.tensor(player_B_y)[encode_length-1:encode_length].to(device).view(1,-1)

            shot_type_predict = np.zeros(batch, dtype = int)
            player_A_x_predict = np.zeros(batch, dtype = np.float32)
            player_A_y_predict = np.zeros(batch, dtype = np.float32)
            player_B_x_predict = np.zeros(batch, dtype = np.float32)
            player_B_y_predict = np.zeros(batch, dtype = np.float32)
            player_predict = np.ones(batch, dtype = int)
            for i in range(player[-1] == 2,len(player_predict),2):
                player_predict[i] = 2
            
            first = True
            for sequence_index in range(encode_length, batch+encode_length):
                predict_xy, predict_shot_type_logit, adjacency_matrix, decoder_node_embedding, original_embedding = decoder(decoder_player, sequence_index+1, decoder_node_embedding, original_embedding, adjacency_matrix, 
                                                                                                                        decoder_player_A_x, decoder_player_A_y, decoder_player_B_x, decoder_player_B_y,
                                                                                                                        shot_type=None, train=False, first=first)
                # sampling
                shot_prob = F.softmax(predict_shot_type_logit, dim=-1)
                while True:
                    output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)
                    if output_shot[0][0] not in [0,1,10]: # 0 is padding, 1, 10 is 發短/長球
                        break
                shot = output_shot.item()

                predict_A_xy = predict_xy[:, 0, :]
                predict_B_xy = predict_xy[:, 1, :]

                #### GMM ####
                epsilon = 1e-20
                predict_A_xy[predict_A_xy==0] = epsilon
                predict_B_xy[predict_B_xy==0] = epsilon

                """
                land GMM loss
                """
                predictA_land_gmm = D.MixtureSameFamily(D.Categorical(F.softmax(predict_A_xy[:,:GMM_NUM], dim=-1)),\
                D.Independent(D.Normal(torch.reshape(predict_A_xy[:,GMM_NUM:GMM_NUM*3],(-1,GMM_NUM,2)),\
                torch.reshape(torch.abs(torch.clamp(predict_A_xy[:,GMM_NUM*3:],-0.1,0.1)),(-1,GMM_NUM,2))), 1))
                predict_A_xy = predictA_land_gmm.sample((1,))
                mean_A = predict_A_xy[0]

                predictB_land_gmm = D.MixtureSameFamily(D.Categorical(F.softmax(predict_B_xy[:,:GMM_NUM], dim=-1)),\
                D.Independent(D.Normal(torch.reshape(predict_B_xy[:,GMM_NUM:GMM_NUM*3],(-1,GMM_NUM,2)),\
                torch.reshape(torch.abs(torch.clamp(predict_B_xy[:,GMM_NUM*3:],-0.1,0.1)),(-1,GMM_NUM,2))), 1))
                predict_B_xy = predictB_land_gmm.sample((1,))
                mean_B = predict_B_xy[0]
                ########    
                
                #sx = torch.exp(predict_A_xy[:, -1, 2]) #sx
                #sy = torch.exp(predict_A_xy[:, -1, 3]) #sy
                #corr = torch.tanh(predict_A_xy[:, -1, 4]) #corr                
#
                #cov = torch.zeros(2, 2).to(device)
                #cov[0, 0]= sx * sx
                #cov[0, 1]= corr * sx * sy
                #cov[1, 0]= corr * sx * sy
                #cov[1, 1]= sy * sy
                #mean = predict_A_xy[:, -1, 0:2]
                #
                #mean_A = mean.cpu().detach().numpy()
                ##mean_A[0][0] = mean_A[0][0] * std_x + mean_x 
                ##mean_A[0][1] = 960 - (mean_A[0][1] * std_y + mean_y )
                #cov_A = cov.cpu().detach().numpy()
                #cov_A = cov_A * std_x * std_y
#
                #all_mean_A.append(mean_A)            
                #all_cov_A.append(cov_A)
#
                #sx = torch.exp(predict_B_xy[:, -1, 2]) #sx
                #sy = torch.exp(predict_B_xy[:, -1, 3]) #sy
                #corr = torch.tanh(predict_B_xy[:, -1, 4]) #corr                
#
                #cov = torch.zeros(2, 2).to(device)
                #cov[0, 0]= sx * sx
                #cov[0, 1]= corr * sx * sy
                #cov[1, 0]= corr * sx * sy
                #cov[1, 1]= sy * sy
                #mean = predict_B_xy[:, -1, 0:2]
                #
                #mean_B = mean.cpu().detach().numpy()
                ##mean_B[0][0] = mean_B[0][0] * std_x + mean_x 
                ##mean_B[0][1] = 960 - (mean_B[0][1] * std_y + mean_y )
                #cov_B = cov.cpu().detach().numpy()
                #cov_B = cov_B * std_x * std_y
#
                #all_mean_B.append(mean_B)
                #all_cov_B.append(cov_B)             

                decoder_player_A_x = predict_A_xy[:, 0, 0:1]
                decoder_player_A_y = predict_A_xy[:, 0, 1:2]
                decoder_player_B_x = predict_B_xy[:, 0, 0:1]
                decoder_player_B_y = predict_B_xy[:, 0, 1:2]

                weights = predict_shot_type_logit[0, 1:]
                weights = F.softmax(weights, dim=0).cpu().detach().numpy()
                #shot_type_record.append(weights)
                first = False

                #player_A_x_record.append(mean_A[0][0])
                #player_A_y_record.append(mean_A[0][1])
                #player_B_x_record.append(mean_B[0][0])
                #player_B_y_record.append(mean_B[0][1])

                shot_type_predict [sequence_index-encode_length] = shot
                player_A_x_predict[sequence_index-encode_length] = mean_A[0][0]
                player_A_y_predict[sequence_index-encode_length] = mean_A[0][1]
                player_B_x_predict[sequence_index-encode_length] = mean_B[0][0]
                player_B_y_predict[sequence_index-encode_length] = mean_B[0][1]

            # swap player
            #if player[0] == 2:
            #    player_A_x_record, player_B_x_record = player_B_x_record, player_A_x_record
            #    player_A_y_record, player_B_y_record = player_B_y_record, player_A_y_record
#
            #    #player_A_x, player_B_x = player_B_x, player_A_x
            #    #player_A_y, player_B_y = player_B_y, player_A_y

            player = np.append(player, player_predict)
            shot_type = np.append(shot_type, shot_type_predict)
            player_A_x = np.append(player_A_x, player_A_x_predict)
            player_A_y = np.append(player_A_y, player_A_y_predict)
            player_B_x = np.append(player_B_x, player_B_x_predict)
            player_B_y = np.append(player_B_y, player_B_y_predict)
            # all_player_A_x_record.append(player_A_x_record)
            # all_player_A_y_record.append(player_A_y_record)
            # all_player_B_x_record.append(player_B_x_record)
            # all_player_B_y_record.append(player_B_y_record)
            # all_shot_type_record.append(shot_type_record)

    return player, shot_type, player_A_x, player_A_y, player_B_x ,player_B_y

# convert multivariate normal distribution to probability for region 1 to 10
def mvn2prob(mean:np.ndarray, cov:np.ndarray):
    # Create a multivariate normal distribution
    mvn = multivariate_normal(mean, cov)

    mean_x = 175.
    mean_y = 467.
    std_x = 82.
    std_y = 192.
    # min_x, max_x, min_y, max_y
    # 1 2 3
    # 4 5 6
    # 7 8 9
    region_bound = [(50,135,150,260),(135,215,150,260),(215,305,150,260),
                    (50,135,260,370),(135,215,260,370),(215,305,260,370),
                    (50,135,370,480),(135,215,370,480),(215,305,370,480),]
    
    prob = np.zeros(10, dtype=np.float32)
    for i, (min_x, max_x, min_y, max_y) in enumerate(region_bound):
        min_x = (min_x - mean_x)/std_x
        max_x = (max_x - mean_x)/std_x
        min_y = (min_y - mean_y)/std_y
        max_y = (max_y - mean_y)/std_y

        # Create a grid of points within the specified region
        x, y = np.mgrid[min_x:max_x:100j, min_y:max_y:100j]

        # Calculate the PDF values for each point in the grid
        pdf_values = mvn.pdf(np.dstack((x, y)))

        # Integrate the PDF values over the region
        prob[i] = np.trapz(np.trapz(pdf_values, x[:, 0], axis=1), y[0, :])
    prob[9] = 1 - np.sum(prob[0:9])

    return prob


def predictAgent(data, encoder, decoder, args, device="cpu"):
    encode_length = args['encode_length']
    GMM_NUM = args['num_GMM']
    encoder.eval(), decoder.eval()

    player, shot_type, player_A_x, player_A_y, player_B_x, player_B_y = data

    max_length = args['max_length']
    player = player[:max_length-2]
    shot_type = shot_type[:max_length-2]
    player_A_x = player_A_x[:max_length-2]
    player_A_y = player_A_y[:max_length-2]
    player_B_x = player_B_x[:max_length-2]
    player_A_y = player_A_y[:max_length-2]
    
    mean_x, std_x = 175., 82.
    mean_y, std_y = 467., 192.

    with torch.no_grad():            
        encoder_player = torch.tensor(player)[-2:].to(device).view(1,-1)
        encoder_shot_type = torch.tensor(shot_type)[-encode_length+1:].to(device).view(1,-1)
        encoder_player_A_x = torch.tensor(player_A_x)[-encode_length:].to(device).view(1,-1)
        encoder_player_A_y = torch.tensor(player_A_y)[-encode_length:].to(device).view(1,-1)
        encoder_player_B_x = torch.tensor(player_B_x)[-encode_length:].to(device).view(1,-1)
        encoder_player_B_y = torch.tensor(player_B_y)[-encode_length:].to(device).view(1,-1)

        encode_node_embedding, original_embedding, adjacency_matrix = encoder(encoder_player, encoder_shot_type, encoder_player_A_x, encoder_player_A_y, encoder_player_B_x, encoder_player_B_y, encode_length)
            
        decoder_node_embedding = encode_node_embedding.clone()
                    
        decoder_player = torch.tensor(player)[0:2].to(device).view(1,-1)

        decoder_player_A_x = torch.tensor(player_A_x)[encode_length-1:encode_length].to(device).view(1,-1)
        decoder_player_A_y = torch.tensor(player_A_y)[encode_length-1:encode_length].to(device).view(1,-1)
        decoder_player_B_x = torch.tensor(player_B_x)[encode_length-1:encode_length].to(device).view(1,-1)
        decoder_player_B_y = torch.tensor(player_B_y)[encode_length-1:encode_length].to(device).view(1,-1)
        
        predict_xy, predict_shot_type_logit, adjacency_matrix, decoder_node_embedding, original_embedding = decoder(decoder_player, encode_length+1, decoder_node_embedding, original_embedding, adjacency_matrix, 
                                                                                                                decoder_player_A_x, decoder_player_A_y, decoder_player_B_x, decoder_player_B_y,
                                                                                                                shot_type=None, train=False, first=True)
        # sampling
        shot_prob = F.softmax(predict_shot_type_logit, dim=-1)
        while True:
            output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)
            if output_shot[0][0] not in [0,1,10]: # 0 is padding, 1, 10 is 發短/長球
                break
        shot = output_shot.item()

        predict_A_xy = predict_xy[:, 0, :]
        predict_B_xy = predict_xy[:, 1, :]

        #### GMM ####
        epsilon = 1e-20
        predict_A_xy[predict_A_xy==0] = epsilon
        predict_B_xy[predict_B_xy==0] = epsilon

        """
        land GMM loss
        """
        weight_A = F.softmax(predict_A_xy[:,:GMM_NUM], dim=-1)
        means_A = torch.reshape(predict_A_xy[:,GMM_NUM:GMM_NUM*3],(-1,GMM_NUM,2))
        stdevs_A = torch.reshape(torch.abs(torch.clamp(predict_A_xy[:,GMM_NUM*3:],-0.3,0.3)),(-1,GMM_NUM,2))
        gmm_param_A = (weight_A.tolist(), means_A.tolist(), stdevs_A.tolist())
        predictA_land_gmm = D.MixtureSameFamily(D.Categorical(weight_A), D.Independent(D.Normal(means_A,stdevs_A), 1))
        predict_A_xy = predictA_land_gmm.sample((1,))
        mean_A = predict_A_xy[0]

        weight_B = F.softmax(predict_B_xy[:,:GMM_NUM], dim=-1)
        means_B = torch.reshape(predict_B_xy[:,GMM_NUM:GMM_NUM*3],(-1,GMM_NUM,2))
        stdevs_B = torch.reshape(torch.abs(torch.clamp(predict_B_xy[:,GMM_NUM*3:],-0.3,0.3)),(-1,GMM_NUM,2))
        gmm_param_B = (weight_B.tolist(), means_B.tolist(), stdevs_B.tolist())
        predictB_land_gmm = D.MixtureSameFamily(D.Categorical(weight_B), D.Independent(D.Normal(means_B,stdevs_B), 1))
        predict_B_xy = predictB_land_gmm.sample((1,))
        mean_B = predict_B_xy[0]
        ########    
        
        #sx = torch.exp(predict_A_xy[:, -1, 2]) #sx
        #sy = torch.exp(predict_A_xy[:, -1, 3]) #sy
        #corr = torch.tanh(predict_A_xy[:, -1, 4]) #corr                
#
        #cov = torch.zeros(2, 2).to(device)
        #cov[0, 0]= sx * sx
        #cov[0, 1]= corr * sx * sy
        #cov[1, 0]= corr * sx * sy
        #cov[1, 1]= sy * sy
        #mean = predict_A_xy[:, -1, 0:2]
        #
        #mean_A = mean.cpu().detach().numpy()
        #cov_A = cov.cpu().detach().numpy()
        #cov_A = cov_A * std_x * std_y
        #prob_A = mvn2prob(mean_A[0], cov_A)
#
        #sx = torch.exp(predict_B_xy[:, -1, 2]) #sx
        #sy = torch.exp(predict_B_xy[:, -1, 3]) #sy
        #corr = torch.tanh(predict_B_xy[:, -1, 4]) #corr                
#
        #cov = torch.zeros(2, 2).to(device)
        #cov[0, 0]= sx * sx
        #cov[0, 1]= corr * sx * sy
        #cov[1, 0]= corr * sx * sy
        #cov[1, 1]= sy * sy
        #mean = predict_B_xy[:, -1, 0:2]
        #
        #mean_B = mean.cpu().detach().numpy()
        #cov_B = cov.cpu().detach().numpy()
        #cov_B = cov_B * std_x * std_y
        #prob_B = mvn2prob(mean_B[0], cov_B)
            
    return shot, mean_A[0][0], mean_A[0][1], mean_B[0][0], mean_B[0][1], gmm_param_A, gmm_param_B


def save(encoder, decoder, args):
    output_folder_name = args['model_folder']
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)
    
    torch.save(encoder.state_dict(), output_folder_name + '/encoder')
    torch.save(decoder.state_dict(), output_folder_name + '/decoder')
