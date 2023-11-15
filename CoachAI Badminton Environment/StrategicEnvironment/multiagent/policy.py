import numpy as np
from pyglet.window import key
import random
import pandas as pd
import ast

# individual agent policy
class Policy(object):
    def __init__(self, env, agent_index):
        super(Policy, self).__init__()
        self.env = env
        # hard-coded keyboard events
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]
       
        
    def action(self, obs):
        if self.env.discrete_action_input:
            u = np.random.randint(4,size=1)+1
            
        else:
            u = np.random.random (5)# 5-d because of no-move action

            
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])



# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication
class InteractivePolicy(Policy):
    def __init__(self, env, agent_index):
        #super(InteractivePolicy, self).__init__(self, env, agent_index)
        self.env = env
        # hard-coded keyboard events
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]
        # register keyboard events with this environment's window
        env.viewers[agent_index].window.on_key_press = self.key_press
        env.viewers[agent_index].window.on_key_release = self.key_release

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_input:
            u = 0
            if self.move[0]: u = 1
            if self.move[1]: u = 2
            if self.move[2]: u = 4
            if self.move[3]: u = 3
        else:
            u = np.zeros(5) # 5-d because of no-move action
            if self.move[0]: u[1] += 5
            if self.move[1]: u[2] += 5
            if self.move[3]: u[3] += 5
            if self.move[2]: u[4] += 5
            if True not in self.move:
                u[0] += 5
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

    # keyboard event callbacks
    def key_press(self, k, mod):
        if k==key.RIGHT:  self.move[0] = True
        if k==key.LEFT: self.move[1] = True
        if k==key.DOWN:    self.move[2] = True
        if k==key.UP:  self.move[3] = True
    def key_release(self, k, mod):
        if k==key.RIGHT:  self.move[0] = False
        if k==key.LEFT: self.move[1] = False
        if k==key.DOWN:     self.move[2] = False
        if k==key.UP:  self.move[3] = False

class Player_Policy(Policy):
    def __init__(self, env, agent_index, matchFilePath, agent_name =None):
        super().__init__(env, agent_index)
        self.env = env
        self.match_index = 0
        #self.match = pd.read_csv(r'Data\match\An_Se_Young_Pornpawee_Chochuwong_TOYOTA_THAILAND_OPEN_2021_QuarterFinals\set1.csv')
        self.match = pd.read_csv(matchFilePath)
        #self.match = pd.read_csv(r'dataExperiment\act_set1_slice.csv')
        #self.__read_multi_match()

        self.player_location_area = -1
        self.landing_pos = -1
        self.agent_index = agent_index
        self.agent_name = agent_name
        self.server = False

        self.ball_type_table = pd.read_excel(r'StrategicEnvironment\Data\ball_type_table.xlsx')
        
        self.rally = 0

        
        self.n = env.world.resistance_force_n = 1
        self.g = env.world.gravitational_acceration = 9.8
        self.V_T = env.world.ball.V_T
    

    
    def jump(self, jump_idx=0):
        self.match_index = jump_idx

        if self.match.loc[self.match_index,'obs_player'] == self.agent_name:
            next_type = self.match.loc[self.match_index,'act_ball_type']
            next_type_name = next_type
            next_type = self.ball_type_table[self.ball_type_table['chinese']==next_type].index[0]
            landing_x = self.match.loc[self.match_index, 'act_landing_location_x']
            landing_y = self.match.loc[self.match_index, 'act_landing_location_y']
            landing_height = 0
            player_x = self.match.loc[self.match_index, 'act_player_defend_x']
            player_y = self.match.loc[self.match_index, 'act_player_defend_y']

            landing_distribution = {
                'x': self.match.loc[self.match_index, 'landing_x'],
                'y': self.match.loc[self.match_index, 'landing_y'],
                'sx': self.match.loc[self.match_index, 'landing_sx'],
                'sy': self.match.loc[self.match_index, 'landing_sy'],
                'tho': self.match.loc[self.match_index, 'landing_tho'],
            }

            shot_distribution = ast.literal_eval(self.match.loc[self.match_index, 'shot_prob'])

        else:
            next_type = None 
            next_type_name = None
            landing_x = None
            landing_y = None
            landing_area = None
            landing_height = None
            player_x = None
            player_y = None
            landing_distribution = None
            shot_distribution = None
            #player_area = None
            #if self.match.loc[self.match_index+1,'obs_serve'] == False:
            #    player_x = self.match.loc[self.match_index+1,'act_hit_x']
            #    player_y = self.match.loc[self.match_index+1,'act_hit_y']
            #    player_area = self.match.loc[self.match_index+1,'hit_area']
            #else:
                
        self.match_index +=1
       
        
        return {
            'player_location':np.array([player_x,player_y]),
            #'player_area':player_area,
            'landing_location':np.array([landing_x,landing_y]),
            #'landing_area':landing_area,
            'ball_type':next_type,
            'ball_type_name': next_type_name,
            'launching_height':None,
            'landing_height':landing_height,
            'landing_distribution':landing_distribution,
            'shot_prob': shot_distribution
        }
        pass

    def action(self, obs):

        ## action space[target_pos(1~9),shoot_ball_to(1~9)]
        #print('state:',obs)

        action = self.match_action(obs)

        return action
        
        if obs['player'] == self.agent_index:
            return self.return_to_mid(obs)
        else:
            return self.random_action()
        obs = obs[self.agent_index]
        ball = obs['ball'].ball.with_palyer

        
        if obs['server'] == 1 and ball.with_player==self.agent_index:
            self.player_location_area=np.random.randint(9,size=1)
            self.landing_pos=np.random.randint(9,size=1)

        else:
            self.player_location_area = obs['player_location_area']
            self.landing_pos=obs['landing_pos']

        return [self.player_location_area,self.landing_pos]
    def fixed_action(self,obs):
        return self. __forehand_clear(obs)
    def __forehand_clear(self, obs):
        
        player_area = 9
        landing_area = 9
        play_type = None
        angle = None
        speed = None
        launching_height = None

            
        if obs['player'] == self.agent_index:
            if obs['server'] == 1:
                play_type = 17
                angle = 30
                speed = 20
                launching_height = 115/2
            else:
                play_type = 11
                angle = 30
                speed = 23
        

            angle = angle/360*2*np.pi

        return {
            'agent_id':self.agent_index,
            'landing_area':landing_area,
            'ball_type':play_type,
            'init_speed':speed,
            'player_area':player_area,
            'launching_height':launching_height
        }
    
    def __forehand_drop(self, obs):
         
        player_area = 9
        landing_area = 9
        play_type = None
        launching_height = None
        if obs['server'] == 1:
            self.server = self.agent_index
        if obs['player'] == self.agent_index:
            if obs['server'] == 1:
                play_type = 17
                launching_height = 115/2
                player_area = 7
               
            else:
                # 挑球
                if self.server == self.agent_index :
                    play_type = 0
                    player_area = 7
                    landing_area = 9
                # 切球
                else:
                    play_type = 15
                    player_area = 9
                    landing_area = 7
        else:
            if self.server:
                player_area = 7
            else:
                player_area = 9
                
        

           

        return {
            'agent_id':self.agent_index,
            'landing_area':landing_area,
            'ball_type':play_type,
            
            'player_area':player_area,
            'launching_height':launching_height
        }
           
    def match_action(self, obs):
        self.rally = obs['match']['rally']
        self.ball_round = obs['match']['ball_round']
        
        if obs['ball']['server'] == True:
            if obs['ball']['player'] == 0:
                player='A'
            elif obs['ball']['player'] == 1:
                player='B'
            while self.match.loc[self.match_index,'obs_serve'] != True or self.match.loc[self.match_index,'obs_player'] != player:
                self.match_index+=1


        print('matchIndex',self.match_index)

        if self.match.loc[self.match_index,'obs_player'] == self.agent_name and self.match.loc[self.match_index,'obs_ball_round']==obs['match']['ball_round']:
            # current ball is last ball
            if self.match.loc[self.match_index,'obs_ball_round'] != self.match.loc[self.match_index+1,'obs_ball_round']:
                last_ball = True
            else:
                last_ball = False
            next_type = self.match.loc[self.match_index,'act_ball_type']
            next_type_name = next_type
            next_type = self.ball_type_table[self.ball_type_table['chinese']==next_type].index[0]
            landing_x = self.match.loc[self.match_index, 'act_landing_location_x']
            landing_y = self.match.loc[self.match_index, 'act_landing_location_y']
            landing_height = 0
            player_x = self.match.loc[self.match_index, 'act_player_defend_x']
            player_y = self.match.loc[self.match_index, 'act_player_defend_y']
            rally = self.match.loc[self.match_index, 'rally']
            round = self.match.loc[self.match_index, 'obs_ball_round']
            state = ast.literal_eval(self.match.loc[self.match_index, 'state'])
            if round > 1:
                last_type = self.match.loc[self.match_index-1,'act_ball_type']
                last_type_name = last_type
                last_type = self.ball_type_table[self.ball_type_table['chinese']==last_type]['english'].iloc[0]
            else:
                last_type = 'receiving'
            #player_region = self.match.loc[self.match_index, 'player_region']
            #opponent_region = self.match.loc[self.match_index, 'opponent_region']
            #landing_region = self.match.loc[self.match_index, 'landing_region']

            #for old continuous ShuttleNet distribution
            #landing_distribution = {
            #    'x': self.match.loc[self.match_index, 'landing_x'],
            #    'y': self.match.loc[self.match_index, 'landing_y'],
            #    'sx': self.match.loc[self.match_index, 'landing_sx'],
            #    'sy': self.match.loc[self.match_index, 'landing_sy'],
            #    'tho': self.match.loc[self.match_index, 'landing_tho'],
            #}

            shot_distribution = ast.literal_eval(self.match.loc[self.match_index, 'shot_prob'])
            landing_region_distribution = ast.literal_eval(self.match.loc[self.match_index, 'land_prob'])
            move_region_distribution = ast.literal_eval(self.match.loc[self.match_index, 'move_prob'])
            # if self.match.loc[self.match_index, 'server'] == 3:
            #    landing_height = 0
            #landing_area = self.match.loc[self.match_index, 'landing_area']
            #if self.match.loc[self.match_index+1,'obs_serve'] == False:
            #    player_x = self.match.loc[self.match_index+1,'act_player_location_x']
            #    player_y = -self.match.loc[self.match_index+1,'opponent_location_y']
            #    #player_area = self.match.loc[self.match_index+1,'opponent_location_area']
            #else:
            #    player_x = None
            #    player_y = None
            #    #landing_area = False
        else:
            last_ball = None
            next_type = None 
            next_type_name = None
            landing_x = None
            landing_y = None
            landing_area = None
            landing_height = None
            player_x = None
            player_y = None
            state = None
            #landing_distribution = None
            shot_distribution = None
            landing_region_distribution = None
            move_region_distribution = None
            rally = None
            round = None
            last_type = None
            #player_region = None
            #opponent_region = None
            #landing_region = None
            #player_area = None
            #if self.match.loc[self.match_index+1,'obs_serve'] == False:
            #    player_x = self.match.loc[self.match_index+1,'act_hit_x']
            #    player_y = self.match.loc[self.match_index+1,'act_hit_y']
            #    player_area = self.match.loc[self.match_index+1,'hit_area']
            #else:
                
        self.match_index +=1
       
        
        return {
            'player_location':np.array([player_x,player_y]),
            #'player_area':player_area,
            'landing_location':np.array([landing_x,landing_y]),
            #'landing_area':landing_area,
            'ball_type':next_type,
            'ball_type_name': next_type_name,
            'launching_height':None,
            'landing_height':landing_height,
            #'landing_distribution':landing_distribution,
            'shot_prob': shot_distribution,
            'land_prob': landing_region_distribution,
            'move_prob': move_region_distribution,
            'rally' : rally,
            'ball_round': round,
            'state': state,
            'last_ball': last_ball,
            'last_type': last_type,
            #'player_region': player_region,
            #'opponent_region': opponent_region,
            #'landing_region': landing_region
        }
    def random_action(self):
        act={
            'player_next_position':np.random.randint(9,size=1),
            'landing_area':None,
            'next_type':None
        }
        return act
    def return_to_mid(self,obs):
        if obs['server']==1:
            next_type = random.choice([17])
            landing_area = self.ball_legal_landing_area[self.ball_type_num_to_word[next_type]][obs['score'][self.agent_index]%2]
        else:
            next_type = 11
            while next_type==16 or next_type==17:
                next_type = np.random.randint(18,size=1)
            landing_area = self.ball_legal_landing_area[self.ball_type_num_to_word[next_type]]
            landing_area - np.random.choice(landing_area)
        act={
            'player_next_position':8,
            'landing_area':landing_area,
            'next_type':next_type
        }
        return act
    def __read_multi_match(self):
        import os 
        root = 'Data\match'
        files = os.listdir(root)
        for f in files:

            set = os.listdir(os.path.join(root,f))
            for s in set:
            
                #print(self.match)
                
                self.match = pd.concat([self.match, pd.read_csv(os.path.join(root,f,s))],ignore_index=True)
    
    
    
    #def get_init_speed_and_angle(self):



