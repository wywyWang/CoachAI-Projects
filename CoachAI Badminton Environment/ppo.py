import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
from RLEnvironment import Env
from RLModel import RLModel
from tqdm import tqdm

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, hidden_width):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, hidden_width),
                            nn.Tanh(),
                            nn.Linear(hidden_width, hidden_width),
                            nn.Tanh(),
                            nn.Linear(hidden_width, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, hidden_width),
                            nn.Tanh(),
                            nn.Linear(hidden_width, hidden_width),
                            nn.Tanh(),
                            nn.Linear(hidden_width, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # shot actor
        self.shot_actor = nn.Sequential(
                nn.Linear(state_dim, hidden_width),
                nn.Tanh(),
                nn.Linear(hidden_width, hidden_width),
                nn.Tanh(),
                nn.Linear(hidden_width, hidden_width),
                nn.Tanh()
            )
        
        self.shot = nn.Sequential(
                nn.Linear(hidden_width, 11),  
                nn.Softmax(dim=-1)
            ) # output shot
        
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, hidden_width), 
                        nn.Tanh(),
                        nn.Linear(hidden_width, hidden_width),
                        nn.Tanh(),
                        nn.Linear(hidden_width, 1)
                    )
        
    def forward(self, state):
        x = self.shot_actor(state)
        shot = self.shot(x)
        return shot
    
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")
    
    def act(self, state, isLaunch):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)

            shot_probs = self.forward(state)
            shot_dist = Categorical(shot_probs)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        shot = shot_dist.sample()
        shot_logprob = shot_dist.log_prob(shot)

        if isLaunch:
            while shot not in [0,9]:
                shot = shot_dist.sample()
        else:
            while shot in [0,9]:
                shot = shot_dist.sample()

        state_val = self.critic(state)

        shot = shot.view(-1)
        action = action.view(-1)
        act = torch.cat((shot, action), dim=0)
        return act, torch.tensor([shot_logprob.detach(), action_logprob.detach()]), state_val.detach(), shot_probs.tolist()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            shot_probs = self.forward(state)
            shot_dist = Categorical(shot_probs)
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action[:, 1:])
        dist_entropy = dist.entropy()

        shot_logprobs = shot_dist.log_prob(action[:, 0])
        shot_dist_entropy = shot_dist.entropy()

        state_values = self.critic(state)
        
        return (shot_logprobs, action_logprobs), state_values, (shot_dist_entropy, dist_entropy)
    
    def state_dict(self):
        return self.actor.state_dict(), self.critic.state_dict()
    
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict[0])
        self.critic.load_state_dict(state_dict[1])

################# PPO hyperparameters ################
update_timestep = 1000 * 4      # update policy every n timesteps
K_epochs = 80               # update policy for K epochs in one PPO update
#
eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor
#
lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network
#
random_seed = 0         # set random seed if required (0 = no random seed)
######################################################
#
state_dim = 6
action_dim = 4
action_std_init = 0.6
# testing_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space=True, train_step=max_rally_count/2)
class PPO(RLModel):
    def __init__(self):

        self.has_continuous_action_space = True

        if self.has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.env = Env()
        self.total_steps = 0
        self.buffer = RolloutBuffer()
        self.hidden_width = 256
        # action_std_init=0.6

        self.policy = ActorCritic(state_dim, action_dim, self.has_continuous_action_space, action_std_init, self.hidden_width).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, self.has_continuous_action_space, action_std_init, self.hidden_width).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.max_ep_len = 1000
        # self.max_train_step = train_step
        self.total_steps = 0
        self.start = 0
        self.ep_steps = 0
        self.episode_reward = 0
        self.total_reward = []
        self.isTrain = False
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, isLaunch):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val, action_probs = self.policy_old.act(state, isLaunch)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten(), action_probs
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

        
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
    
    def train(self, state, is_launch):
        state = state[-1]
        with torch.no_grad():
            # state = torch.FloatTensor(state).to(device)
            state = np.array(state).reshape(6,).tolist()
            state = self.normalize(state)
            _action, action_probs = self.select_action(state, is_launch)
            act = np.array(_action[1:5]).reshape(2, 2)

            # state = list(map(int, state))
            # shotlist, movelist, landlist = action_prob
            # shotlist, movelist, landlist = self.action_findtuned(state, shotlist, movelist, landlist)

            # shot_prob = action_probs[0]
            
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

        if self.isTrain:
            self.ep_steps += 1
            self.total_steps += 1
            done = False
            next_state, reward = self.env.step(action, is_launch)
                
            if reward == -1 or self.ep_steps >= self.max_ep_len:
                done = True
                
            self.episode_reward += reward
            # saving reward and is_terminals
            self.buffer.rewards.append(reward)
            self.buffer.is_terminals.append(done)

            if self.total_steps % 100 == 0:
                self.update()

            if self.total_steps % 50000 == 0:
                print("--------------------------------------------------------------------------------------------")
                print("Steps:", self.total_steps, " avg reward(per rally):", float(sum(self.total_reward)/len(self.total_reward)))
                np.save(f'./data_train/PPO_reward_from_{self.start}.npy', np.array(self.total_reward))
                checkpoint_path = f'./data_train/PPO/PPO_{self.total_steps}.pth'
                self.save_(checkpoint_path)
                print("model saved")
                print("--------------------------------------------------------------------------------------------")

            if done:
                self.ep_steps = 0
                self.total_reward.append(self.episode_reward)
                self.episode_reward = 0

        return action, (action_probs, [], [])
    
    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs_, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            logprobs = torch.stack([torch.stack(items) for items in zip(*logprobs_)])
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios.T[0] * advantages
            surr2 = torch.clamp(ratios.T[0], 1-self.eps_clip, 1+self.eps_clip) * advantages

            lsurr1 = ratios.T[1] * advantages
            lsurr2 = torch.clamp(ratios.T[1], 1-self.eps_clip, 1+self.eps_clip) * advantages


            # final loss of clipped objective PPO
            # loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            loss = -torch.min(surr1, surr2) + -torch.min(lsurr1, lsurr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * (dist_entropy[0]+dist_entropy[1])

            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save_(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       

