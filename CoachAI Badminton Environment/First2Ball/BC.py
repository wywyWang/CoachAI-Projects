import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os
from tqdm import tqdm
import argparse
import pandas as pd
import torch.distributions.multivariate_normal as torchdist


class replay_buffer():

    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, data, result):
        self.memory.append([data, result])

    def sample(self, batch_size):

        batch = random.sample(self.memory, batch_size)
        data, result = zip(*batch)
        return data, result


class Net(nn.Module):

    def __init__(self, player_pos=2, opponent_pos=2, landing_pos=2, shot_dim=64, player_dim=64, opponent_dim=64, ball_dim=64, hidden_layer_size=512):
        super(Net, self).__init__()

        shot_type_count = 12
        max_length = 2
        self.input_state = shot_dim+player_dim+opponent_dim + \
            ball_dim  # the dimension of state space

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.type_embedding = nn.Embedding(shot_type_count, shot_dim)
        self.type_linear = nn.Linear(shot_dim*max_length, shot_dim)
        self.player_embedding = nn.Linear(2*max_length, player_dim)
        self.opponent_embedding = nn.Linear(2*max_length, opponent_dim)
        self.ball_embedding = nn.Linear(2*max_length, ball_dim)

        self.input_layer = nn.Linear(self.input_state, 256)  # input layer
        self.fc1 = nn.Linear(256, hidden_layer_size)  # hidden layer
        self.fc2 = nn.Linear(
            hidden_layer_size, hidden_layer_size)  # hidden layer
        self.shot = nn.Linear(
            hidden_layer_size, shot_type_count)  # output layer
        self.land = nn.Linear(hidden_layer_size, 2)
        self.move = nn.Linear(hidden_layer_size, 2)

    def forward(self, player_x, player_y, opponent_x, opponent_y, ball_x, ball_y, type):
        device = self.device

        player = torch.cat((player_x, player_y), dim=1).float()
        opponent = torch.cat((opponent_x, opponent_y), dim=1).float()
        ball = torch.cat((ball_x, ball_y), dim=1).float()

        embedded_player = self.player_embedding(player)
        embedded_opponent = self.opponent_embedding(opponent)
        embedded_ball = self.ball_embedding(ball)
        embedded_type = self.type_embedding(type)
        embedded_type = self.type_linear(embedded_type.view(type.shape[0], -1))

        concated_input = torch.cat(
            (embedded_player, embedded_opponent, embedded_ball, embedded_type), dim=1)

        x = F.relu(self.input_layer(concated_input))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        shot_values = self.shot(x)
        land_values = self.land(x)
        move_values = self.move(x)

        return shot_values, land_values, move_values


class BCAgent:
    def __init__(self, is_train=False, training_data=None, config=None, learning_rate=0.0003, batch_size=128, capacity=120000):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.config = config
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.capacity = capacity

        self.buffer = replay_buffer(self.capacity)

        self.evaluate_net = Net()  # the evaluate network
        self.evaluate_net.to(self.device)
        # self.target_net = Net()  # the target network

        self.optimizer = torch.optim.Adam(
            self.evaluate_net.parameters(), lr=self.learning_rate)

        if is_train:
            grouped = training_data.groupby('rally_id')
            self.insertData(grouped)

    def insertData(self, grouped):
        max_length = self.config['max_length']
        for group_name, rows in tqdm(grouped):
            player_x = [0] * max_length
            player_y = [0] * max_length
            opponent_x = [0] * max_length
            opponent_y = [0] * max_length
            ball_x = [0] * max_length
            ball_y = [0] * max_length
            prev_type = [0] * max_length
            for i in range(0, len(rows)-1):
                next_row = rows.iloc[i+1]
                row = rows.iloc[i]

                # state
                player_x[i] = row['player_location_x']
                player_y[i] = row['player_location_y']
                opponent_x[i] = row['opponent_location_x']
                opponent_y[i] = row['opponent_location_y']
                if i == 0:
                    prev_type[i] = 0
                    ball_x[i] = player_x[i]
                    ball_y[i] = player_y[i]
                else:
                    last_row = rows.iloc[i-1]
                    prev_type[i] = last_row['type']
                    ball_x[i] = -last_row['landing_x']
                    ball_y[i] = -last_row['landing_y']

                # action
                type = row['type']
                landing_x = row['landing_x']
                landing_y = row['landing_y']
                move_x = -next_row['opponent_location_x']
                move_y = -next_row['opponent_location_y']

                given = player_x[:], player_y[:], opponent_x[:
                                                             ], opponent_y[:], ball_x[:], ball_y[:], prev_type[:]
                result = type, landing_x, landing_y, move_x, move_y
                self.buffer.insert(given, result)

    def learn(self):
        device = self.device
        steps = 10_000
        for i in tqdm(range(steps)):
            # action(1~10,1~10,1~9)
            data, result = self.buffer.sample(self.batch_size)

            player_x = torch.tensor([row[0] for row in data]).to(device)
            player_y = torch.tensor([row[1] for row in data]).to(device)
            opponent_x = torch.tensor([row[2] for row in data]).to(device)
            opponent_y = torch.tensor([row[3] for row in data]).to(device)
            ball_x = torch.tensor([row[4] for row in data]).to(device)
            ball_y = torch.tensor([row[5] for row in data]).to(device)
            type = torch.tensor([row[6] for row in data]).long().to(device)

            target_type = torch.tensor([row[0]
                                       for row in result]).long().to(device)
            target_landing_x = torch.tensor(
                [row[1] for row in result]).to(device)
            target_landing_y = torch.tensor(
                [row[2] for row in result]).to(device)
            target_move_x = torch.tensor([row[3] for row in result]).to(device)
            target_move_y = torch.tensor([row[4] for row in result]).to(device)

            target_landing = torch.cat(
                (target_landing_x.unsqueeze(-1), target_landing_y.unsqueeze(-1)), dim=1).float()
            target_move = torch.cat(
                (target_move_x.unsqueeze(-1), target_move_y.unsqueeze(-1)), dim=1).float()

            # for every state, choose specify action
            predict = self.evaluate_net.forward(
                player_x, player_y, opponent_x, opponent_y, ball_x, ball_y, type)

            CE_loss = nn.CrossEntropyLoss(reduction='sum')
            MSE_loss = nn.MSELoss(reduction='sum')
            shot_loss = CE_loss(predict[0], target_type)
            land_loss = MSE_loss(predict[1], target_landing)
            move_loss = MSE_loss(predict[2], target_move)
            loss = shot_loss + land_loss + move_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 2000 == 0:
                tqdm.write(f'[shot CE: {shot_loss.item()/self.batch_size:.4f}] '
                           f'[land MSE: {land_loss.item()/self.batch_size:.4f}] '
                           f'[move MSE: {move_loss.item()/self.batch_size:.4f}]')
        self.save(f'ShuttleDyMFNet2ball-final.pt')
        print(f'[shot CE: {shot_loss.item()/self.batch_size:.4f}] '
              f'[land MSE: {land_loss.item()/self.batch_size:.4f}] '
              f'[move MSE: {move_loss.item()/self.batch_size:.4f}]')

    def predict(self, player, opponent, ball, type):
        device = self.device
        with torch.no_grad():
            player = player.to(device)
            opponent = opponent.to(device)
            type = type.to(device)
            ball = ball.to(device)

            shot, land, move = self.evaluate_net(player[:, 0], player[:, 1],
                                                 opponent[:,
                                                          0], opponent[:, 1],
                                                 ball[:, 0], ball[:, 1], type)

        return shot.cpu(), land.cpu(), move.cpu()

    def save(self, name):
        torch.save(self.evaluate_net.state_dict(), name)

    def load(self, name):
        self.evaluate_net.load_state_dict(torch.load(name))
        self.evaluate_net.eval()
