import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from RLEnvironment import Env
from RLModel import RLModel
from torch.distributions import Categorical
from datetime import datetime
# import tqdm

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

class Critic(nn.Module): 
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim + 1, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)
        
    def forward(self, s, a):
        q = F.relu(self.l1(torch.cat([s, a], 0)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q
    
class A2C(RLModel):
    def __init__(self, state_dim, action_dim, gamma, lr, train_step):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.GAMMA = gamma  # discount factor
        self.lr = lr  # learning rate

        self.isTrain = False
        self.max_ep_len = 1000
        self.max_train_step = train_step
        self.save_num = 0
        self.total_steps = 0
        self.start = 0
        self.episode_reward = 0
        self.total_reward = []
        self.ep_steps = 0
        self.i_episode = 0

        self.env = Env()    
        self.device = torch.device('cuda:0')
        self.actor = Actor(state_dim, action_dim, self.hidden_width).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(state_dim, action_dim, self.hidden_width).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

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

    def learn(self, s, a, r, s_, dw):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float).to(self.device), 0)
        s_ = torch.unsqueeze(torch.tensor(s_, dtype=torch.float).to(self.device), 0)
        # a_flat = np.append(a[0], np.array(a[1]).reshape(4,)[0:4])
        a = torch.unsqueeze(torch.tensor(a, dtype=torch.float).to(self.device), 0)

        with torch.no_grad():  # td_target has no gradient
            shot_probs_target, a_target = self.actor(s_)
            shot_dist_target = Categorical(shot_probs_target)
            shot_target = shot_dist_target.sample()
            action = torch.cat([shot_target, a_target.squeeze(0)], dim=0)
            v_s_ = self.critic(s_.squeeze(0), action)
            td_target = r + self.GAMMA * (1 - dw) * v_s_

        # Update actor
        shot_probs, a_ = self.actor(s)
        shot_dist = Categorical(shot_probs)
        shot = shot_dist.sample()
        action = torch.cat([shot, a_.squeeze(0)], dim=0)
        actor_loss = -self.critic(s.squeeze(0), action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic
        v_s = self.critic(s.squeeze(0), a.squeeze(0))
        critic_loss = self.MseLoss(td_target, v_s)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
    def action(self, state, is_launch):
        with torch.no_grad():
            # state = torch.FloatTensor(state).to(device)
            state = np.array(state).reshape(6,).tolist()
            state = self.normalize(state)
            _action, action_probs = self.choose_action(state)
            # a = np.array([a_[0]+1, np.array(a_[1:5]).reshape(2, 2)], dtype=object)
            act = np.array(_action[1:5]).reshape(2, 2)

            # shot_prob = action_probs.squeeze(0)
            # if isLaunch:
            #     shotsample = np.append(shot_prob[0].cpu(), shot_prob[9].cpu())
            #     shot = np.random.choice([0, 9], p=shotsample/shotsample.sum())
            # else:
            #     shotsample = np.append(shot_prob[1:9].cpu(), shot_prob[10].cpu())
            #     shot = np.random.choice([1,2,3,4,5,6,7,8,10], p=shotsample/shotsample.sum())
            
            # _action[0] = shot
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

            # if self.total_steps % 1000 == 0:
            self.learn(state, _action, r, s_, dw)

            if self.total_steps % 50000 == 0 or self.total_steps == self.max_train_step:
                print("--------------------------------------------------------------------------------------------")
                print("Steps:", self.total_steps, " avg reward(per rally):", float(sum(self.total_reward)/len(self.total_reward)))
                np.save(f'./data_train/A2C_reward_from_{self.start}.npy', np.array(self.total_reward))
                self.save_(self.total_steps)
                print("model saved")
                # print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            if done :
                self.ep_steps = 0
                self.total_reward.append(self.episode_reward)
                self.episode_reward = 0

        return action, (action_probs, [], [])
    
    def save_(self, num):
        torch.save(self.actor.state_dict(), f"./data_train/A2C/A2C_actor_{num}.pth")
        torch.save(self.critic.state_dict(), f"./data_train/A2C/A2C_critic_{num}.pth")

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