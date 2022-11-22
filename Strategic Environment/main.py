from time import sleep
from make_env import make_env
from multiagent.policy import InteractivePolicy, Player_Policy, Policy
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
import numpy as np

# load scenario and bulid world
scenario = scenarios.load('badminton.py').Scenario()
world=scenario.make_world(resistance_force_n = 1)

world.discrete_court = False
world.decide_defend_location = False
world.number_of_match = 3 # total number of match
world.player = 1 # B


#world.ball.returnable_distance = 100
#world.ball.returnable_height = 150

# make environment for current world
env = make_env('badminton')
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, top_viewer = False, side_viewer = False)

# set policy for each agent
env.render()
policies = []
policies.append(Player_Policy(env,0,'A'))
policies.append(Player_Policy(env,1,'B'))

obs_n = env.reset()

def game_end(obs):
    if(obs['score'][0]>=21 or obs['score'][1]>=21):
        return True
    return False

# training process
while True:
    while True:
        while True:
            act_n = []

            for i, policy in enumerate(policies):
                act_n.append(policy.action(obs_n[i]))
            obs_n, reward_n, done_n, info_n = env.step(act_n)
            
            if done_n[0].rally_done:
                env.reset()
                break 
            
        if done_n[0].match_done:
            
            break
    if done_n[0].all_match_done:
        break
env.save_match_data('match_data_1.xlsx')