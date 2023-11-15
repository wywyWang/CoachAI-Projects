import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from RLEnvironment import Env
from RLModel import RLModel
from torch.distributions import Categorical
from datetime import datetime
import tqdm

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, action_dim)
        
        self.shot = nn.Sequential(
                nn.Linear(hidden_width, 11),  
                nn.Softmax(dim=-1)
            ) # output shot
    
    def forward(self, s):
        s = F.relu(self.l1(s))
        s = F.relu(self.l2(s))
        a = torch.tanh(self.l3(s))  # [-max,max]
        shot = torch.tanh(self.shot(s))
        # shot = shot.append(shot, a)
        # return shot, a
        return shot, a


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim + 1, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)
        
    def forward(self, s, a):
        q = F.relu(self.l1(torch.cat([s, a], 1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim):
        self.max_size = int(1e6)
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim+1))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))
        self.device = torch.device('cuda:0')

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float).to(self.device)
        batch_a = torch.tensor(self.a[index], dtype=torch.float).to(self.device)
        batch_r = torch.tensor(self.r[index], dtype=torch.float).to(self.device)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float).to(self.device)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float).to(self.device)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw

    # def clear(self):
    #     del self.s[:]
    #     del self.a[:]
    #     del self.r[:]
    #     del self.s_[:]
    #     del self.dw[:]

class DDPG(RLModel):
    def __init__(self, state_dim, action_dim, gamma, tau, lr, train_step):
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.batch_size = 256  # batch size
        self.GAMMA = gamma  # discount factor
        self.TAU = tau  # Softly update the target network
        self.lr = lr  # learning rate

        self.isTrain = False
        self.max_ep_len = 1000
        self.max_train_step = train_step
        self.save_num = 0
        self.total_steps = 0
        self.start = 0
        self.ep_steps = 0
        self.episode_reward = 0
        self.total_reward = []
        self.i_episode = 0
        self.env = Env()
        self.device = torch.device('cuda:0')
    
        self.actor = Actor(state_dim, action_dim, self.hidden_width).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim, self.hidden_width).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(state_dim, action_dim)
        self.MseLoss = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float).to(self.device), 0)
        shot_probs, a = self.actor(s)
        shot_dist = Categorical(shot_probs)
        shot = shot_dist.sample()
        shot = shot.cpu().numpy()
        act = a.detach().cpu().numpy()
        action = np.append(shot, act)
        return action, shot_probs
    
    def get_shot_batch(self, probs):
        shot_batch = []
        for prob in probs:
            if torch.isnan(prob).any():
                shot_batch.append(float('nan'))
            else:
                shot_dist = Categorical(prob)
                shot = shot_dist.sample()
                shot = shot.item()
                shot_batch.append(shot)
        shot_batch = torch.tensor(shot_batch).to(self.device)
        return shot_batch

    def learn(self, relay_buffer):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  # Sample a batch

        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            shot_probs_target, a_target = self.actor_target(batch_s_)
            shot_batch = self.get_shot_batch(shot_probs_target)
            batch_action = torch.cat((shot_batch.unsqueeze(1), a_target), dim=1)
            Q_ = self.critic_target(batch_s_, batch_action)
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * Q_

        # Compute the current Q and the critic loss
        current_Q = self.critic(batch_s, batch_a)
        critic_loss = self.MseLoss(target_Q, current_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        shot_probs, a = self.actor(batch_s)
        shot_batch = self.get_shot_batch(shot_probs)
        batch_action = torch.cat((shot_batch.unsqueeze(1), a), dim=1)
        actor_loss = -self.critic(batch_s, batch_action).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

    def action(self, state, is_launch):
        with torch.no_grad():
            # state = torch.FloatTensor(state).to(device)
            state = np.array(state).reshape(6,).tolist()
            state = self.normalize(state)
            _action, action_probs = self.choose_action(state)
            act = np.array(_action[1:5]).reshape(2, 2)

            # shot_prob = action_probs.squeeze(0)
            # if isLaunch:
            #     shotsample = np.append(shot_prob[0], shot_prob[9])
            #     shot = np.random.choice([0, 9], p=shotsample/shotsample.sum())
            # else:
            #     shotsample = np.append(shot_prob[1:9], shot_prob[10])
            #     shot = np.random.choice([1,2,3,4,5,6,7,8,10], p=shotsample/shotsample.sum())

            land = act[0]
            move = act[1]
            land = self.std(land)
            move = self.std(move)
            action = (int(_action[0].item())+1, tuple(land.tolist()), tuple(move.tolist()))

        if is_launch:
            self.ep_steps = 0
            self.total_reward.append(self.episode_reward)
            self.episode_reward = 0
        done = False
        if self.isTrain:
            self.ep_steps += 1
            self.total_steps += 1
            s_, r = self.env.step(action, is_launch)
            if s_ is not None:
                s_ = np.concatenate(s_)
                s_ = self.normalize(s_)

            if r == -1 or self.ep_steps >= self.max_ep_len:
                done = True

            self.episode_reward += r
            if done and self.ep_steps != self.max_ep_len:
                dw = True
            else:
                dw = False
            #     evaluate_reward = evaluate_policy(env_evaluate, agent)
            #     evaluate_rewards.append(evaluate_reward)
            self.replay_buffer.store(state, _action, r, s_, dw)  # Store the transition

            # Take 500 steps,then update the networks 10 times
            if self.total_steps % 1000 == 0:
                # for _ in range(10):
                self.learn(self.replay_buffer)

            if self.total_steps % 50000 == 0 or self.total_steps == self.max_train_step:
                print("--------------------------------------------------------------------------------------------")
                print("Steps:", self.total_steps, " avg reward(per rally):", float(sum(self.total_reward)/len(self.total_reward)))
                np.save(f'./data_train_momo/DDPG_reward_from_{self.start}.npy', np.array(self.total_reward))
                self.save_(self.total_steps)
                print("model saved")
                print("--------------------------------------------------------------------------------------------")

            if done:
                self.ep_steps = 0
                self.total_reward.append(self.episode_reward)
                self.episode_reward = 0

        return action, (action_probs, [], [])
    
    def save_(self, num):
        torch.save(self.actor.state_dict(), f"./data_train_momo/DDPG/DDPG_actor_{num}.pth")
        torch.save(self.critic.state_dict(), f"./data_train_momo/DDPG/DDPG_critic_{num}.pth")

    def load_(self, path_a: str, path_c: str):
        self.actor.load_state_dict(torch.load(path_a))
        self.critic.load_state_dict(torch.load(path_c))

    def normalize(self, state):
        for idx, i in enumerate(state):
            if idx % 2 == 0:
                state[idx] = state[idx] / 82.
            else:
                state[idx] = state[idx] / 192.
        return state
    
    def std(self, coord):
        std_x = 82.
        std_y = 192.

        coord[0] = coord[0] * std_x
        coord[1] = coord[1] * std_y

        return coord

