import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils import save
from ShuttleNet.Decoder import ShotGenDecoder
from ShuttleNet.Encoder import ShotGenEncoder

### GMM ###
import torch.nn.functional as F
import torch.distributions as D
####

PAD = 0

def Gaussian2D_loss(V_pred, V_target):
    Nx = V_target[:, 0] - V_pred[:, 0] #𝑥₁-μ₁
    Ny = V_target[:, 1] - V_pred[:, 1] #𝑥₂-μ₂

    Sx  = torch.exp(V_pred[:, 2]) #σ₁
    Sy  = torch.exp(V_pred[:, 3]) #σ₂
    rho = torch.tanh(V_pred[:, 4]) #ρ

    SxSy = Sx * Sy #σ₁σ₂
    NxNy = Nx * Ny #(𝑥₁-μ₁)(𝑥₂-μ₂)

    Sigma = 1 - rho ** 2 #1-ρ²

    '''
                       1             ((𝑥₁-μ₁)/σ₁)²-2ρ(𝑥₁-μ₁)(𝑥₂-μ₂)/σ₁/σ₂+((𝑥₂-μ₂)/σ₂)²
    f(𝑥₁, 𝑥₂) = —————————————— exp[- ——————————————————————————————————————————————————]
                 2π σ₁σ₂ √1-ρ²                             2(1-ρ²)
    '''
    PDF = 1/(2 * np.pi * SxSy * torch.sqrt(Sigma)) * torch.exp(-((Nx/Sx)**2 - 2*rho*(NxNy/SxSy) + (Ny/Sy)**2)/(2*Sigma))

    result = -torch.log(torch.clamp(PDF, min=1e-20))
    output = torch.sum(result)
    return output

def shotGen_trainer(k_fold_index, data_loader, val_data_loader, config, device="cpu"):
    GMM_NUM = config['num_GMM']
    encoder = ShotGenEncoder(config)
    decoder = ShotGenDecoder(config)
    encoder.embedding.areaEmbedding.weight = decoder.embedding.areaEmbedding.weight
    encoder.embedding.shotEmbedding.weight = decoder.embedding.shotEmbedding.weight
    encoder.embedding.playerEmbedding.weight = decoder.embedding.playerEmbedding.weight
    decoder.predictor.player_embedding.weight = decoder.embedding.playerEmbedding.weight
    
    # total model parameters
    #total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) \
    #             + sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    #print(f"Model params: {total_params}")

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config['lr'])
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=config['lr'])

    encoder.to(device)
    decoder.to(device)

    criterion = {
        'entropy': nn.CrossEntropyLoss(ignore_index=0, reduction='sum'),
        'mse': nn.MSELoss(reduction='sum'),
        'mae': nn.L1Loss(reduction='sum')
    }
    for key, value in criterion.items():
        criterion[key].to(device)


    
    encode_length = config['encode_length']
    record_loss = {
        'total': [],
        'shot': [],
        'area': []
    }
    record_val_loss = {
        'total': [],
        'entropy': [],
        'mse': [],
        'mae': []
    }
    best_val_loss = 1e6

    for epoch in tqdm(range(config['epochs']), desc='Epoch: '):
        encoder.train(), decoder.train()
        total_loss, total_shot_loss, total_area_loss = 0, 0, 0
        total_instance = 0

        for item in data_loader:
            batch_input_shot, batch_input_x, batch_input_y, batch_input_player = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device)
            batch_target_shot, batch_target_x, batch_target_y, batch_target_player = item[4].to(device), item[5].to(device), item[6].to(device), item[7].to(device)
            seq_len, seq_sets = item[8].to(device), item[9].to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            #feed first encode_length count of data to encoder
            input_shot   =   batch_input_shot[:, :encode_length]
            input_x      =      batch_input_x[:, :encode_length]
            input_y      =      batch_input_y[:, :encode_length]
            input_player = batch_input_player[:, :encode_length]

            # feed shot, x, y, player to encoder
            # shot = (dataCount, encodeLength) each shot is int and [1, 10]
            # x    = (dataCount, encodeLength) each y is standardlize coord
            # y    = (dataCount, encodeLength) each y is standardlize coord
            #player= (dataCount, encodeLength) each player is {1,2}
            encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player)

            #the rest data
            input_shot    =    batch_input_shot[:, encode_length:]
            input_x       =       batch_input_x[:, encode_length:]
            input_y       =       batch_input_y[:, encode_length:]
            input_player  =  batch_input_player[:, encode_length:]
            target_shot   =   batch_target_shot[:, encode_length:]
            target_x      =      batch_target_x[:, encode_length:]
            target_y      =      batch_target_y[:, encode_length:]
            target_player = batch_target_player[:, encode_length:]
            output_xy, output_shot_logits = decoder(input_shot, input_x, input_y, input_player, encode_local_output, encode_global_A, encode_global_B, target_player)
            
            # mask the info encoder generate
            pad_mask = (input_shot!=PAD)
            output_shot_logits = output_shot_logits[pad_mask]
            target_shot        =        target_shot[pad_mask]
            output_xy          =          output_xy[pad_mask]
            target_x           =           target_x[pad_mask]
            target_y           =           target_y[pad_mask]

            _, output_shot = torch.topk(output_shot_logits, 1)
            gold_xy = torch.cat((target_x.unsqueeze(-1), target_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)

            total_instance += len(target_shot)

            if len(target_shot) == 0:
                continue

            # conver to long to avoid RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Int'
            target_shot = target_shot.type(torch.LongTensor).to(device)
            loss_shot = criterion['entropy'](output_shot_logits, target_shot)
            #loss_area = Gaussian2D_loss(output_xy, gold_xy)

            # using GMM
            epsilon = 1e-20
            output_xy[output_xy==0] = epsilon

            """
            land GMM loss
            """
            land_gmm = D.MixtureSameFamily(D.Categorical(F.softmax(output_xy[:,:GMM_NUM], dim=-1)),\
            D.Independent(D.Normal(torch.reshape(output_xy[:,GMM_NUM:GMM_NUM*3],(-1,GMM_NUM,2)),\
            torch.reshape(torch.abs(torch.clamp(output_xy[:,GMM_NUM*3:],-0.3,0.3)),(-1,GMM_NUM,2))), 1))
            loss_area = -land_gmm.log_prob(gold_xy).sum()

            loss = loss_shot + loss_area
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()
            total_shot_loss += loss_shot.item()
            total_area_loss += loss_area.item()

        total_loss = round(total_loss / total_instance, 4)
        total_shot_loss = round(total_shot_loss / total_instance, 4)
        total_area_loss = round(total_area_loss / total_instance, 4)

        record_loss['total'].append(total_loss)
        record_loss['shot'].append(total_shot_loss)
        record_loss['area'].append(total_area_loss)

    config['total_loss'] = total_loss
    config['total_shot_loss'] = total_shot_loss
    config['total_area_loss'] = total_area_loss
    save(encoder, decoder, config, k_fold_index)

    return record_loss, record_val_loss