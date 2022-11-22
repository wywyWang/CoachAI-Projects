from collections import namedtuple
from unittest import result
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
from multiagent.rendering import Color
import time
import pandas as pd

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, top_viewer=True, side_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        self.match_data = pd.DataFrame(columns=['match', 'rally', 'ball_round']).set_index(['match', 'rally', 'ball_round'])
        



        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)
        self.viewers = [None, None]
        self.top_viewer = top_viewer
        self.side_viewer = side_viewer
        
        #self.record = pd.DataFrame
    
        
        self._reset_render()

    def step(self, action_n):
        obs_n = []
        reward_n = [0, 0]
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        self.ball = self.world.ball
        status = 0

        #print('actionA:',action_n[0])
        #print('actionB:',action_n[1])
        
        self.__action_preprocessed(action_n)
        self.__save_match(action_n)
        
        print('rally:',self.world.rally,'ball round:',self.world.ball_round, 'player:',self.world.player)
        print('actionA:',action_n[0])
        print('actionB:',action_n[1])
        print('\n----------------------------------------------\n')

        #print('pre_step')
        
        while self.world.pre_step(action_n):
            self.time += 1
            self.render()
            if self.side_viewer==True or self.top_viewer==True:
                time.sleep(self.world.time_scale)
        #time.sleep(3)
        #print('pre_step finish')

        #print('step')

        self.world.ball.returned = True

        while status == 0:
            status = self.world.step(action_n)
            self.time += 1
            self.render()
            if self.side_viewer==True or self.top_viewer==True:
                time.sleep(self.world.time_scale)
            if self.serving:
                if self.side_viewer==True or self.top_viewer==True:
                    time.sleep(1)
                self.serving = False
        #print('step finish')
        # Check if this game is finished
        winner = self.__check_done_reason(status=status)
        print('winner:',winner, 'status',status)
        
        if winner is not None:
            reward_n[winner] += 10

            self.world.player = winner
            self.ball.serve = True
            self.world.rally+=1
            self.world.ball_round=1
            self.__update_score(winner)
            if self.side_viewer==True or self.top_viewer==True:
                time.sleep(1)
        else:
            reward_n[self.world.player] += 1
            self.world.player = 1 - self.world.player
            self.world.ball_round+=1
            

        for idx,agent in enumerate(self.agents):
            done_n.append(self._get_done(status=status))
            
        for idx,agent in enumerate(self.agents):
            obs_n.append(self._get_obs(idx))
            info_n['n'].append(self._get_info(agent))
        
        
        return obs_n, reward_n, done_n, info_n

    def step_with_step(self, state, action):
        def __coordinate_transform(agent_index, pos):
            if None in pos:
                return np.array([None, None])
            mid_point = [177.5, 480]
            transformed_pos = pos
            
            if agent_index == 1:
                transformed_pos = -transformed_pos
            transformed_pos+= mid_point
            return transformed_pos

        ball =self.world.ball
        player = self.agents[0]
        opponent = self.agents[1]
        world = self.world

        world.player = 0
        world.match = state['match']['match']
        world.rally = state['match']['rally']
        world.ball_round = state['match']['ball_round']
        world.score = state['match']['score']
        world.time = state['match']['time'] / world.time_scale

        ball.serve = state['match']['serve']
        ball.state.p_pos = __coordinate_transform(0, state['ball']['location'])
        ball.state.p_height = state['ball']['height']
        ball.state.p_vel = state['ball']['velocity']

        player.state.p_pos = __coordinate_transform(0,state['player']['location'])
        player.state.p_vel = state['player']['velocity']

        opponent.state.p_pos =  __coordinate_transform(0,state['opponent']['location'])
        opponent.state.p_vel = state['opponent']['velocity']
        
        action_n = [action]
        action_n.append({
            'player_location':[None, None]
        })

        self.step(action_n)
        

    
    def reset(self):
        #print(self.world.score)
        
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        
        self.agents = self.world.policy_agents
        for idx,agent in enumerate(self.agents):
            obs = self._get_obs(idx)
            obs_n.append(obs)
        self.serving = True
        return obs_n
    
    def save_match_data(self, path):
        
        
        self.match_data.to_excel(path)
        #playerA_match_data = self.match_data.copy()
        #location_column = ['ball_location_x', 'ball_location_y', 'playerA_location_x', 'playerA_location_y','playerB_location_x', 'playerB_location_y']
        #for col in location_column:
        #    playerA_match_data[col] = self.match_data[col].map(lambda x:__coordinate_transform(0, x))
        #print(self.match_data)
       

    
    def __update_score(self, winner):
       
        self.world.score[winner] += 1
        if winner==0:
            self.world.player = 0
        else:
            self.world.player = 1
    
    def __check_done_reason(self, status):
        
        world = self.world
        ind = (world.match, world.rally, world.ball_round)
        lose_reason = None
        winner = None 
        # Return success
        if status==1:
            lose_reason = None 
            winner = None
        # Ball fall 
        if status==2:
            ball = self.ball
            ball_pos = ball.state.p_pos
            ball_height = ball.state.p_height
            # 出界
            if ball_pos[1] < 150 or ball_pos[1] > 810 or ball_pos[0] < 50 or ball_pos[0] >305:
                lose_reason = '出界'
                winner =  1 - world.player
            # 下方球場界內
            elif ball_pos[1] < 480:
                lose_reason = '落地得分'
                winner = 1
            # 上方球場界內
            else:
                lose_reason = '落地得分'
                winner = 0
        # hit net
        if status==3:
            lose_reason = '掛網'
            winner = 1 - world.player
        if status==4:
            lose_reason = '發球失敗'
            #print('發球:',world.player)
            winner = 1-world.player
        if status == 5:
            lose_reason = '回擊失敗'
            winner = 1-world.player

        if winner is not None:
            self.match_data.loc[ind,'server'] = 3
            if winner == 0:
                self.match_data.loc[ind,'winner'] = 'A'
            else:
                self.match_data.loc[ind,'winner'] = 'B'
        
            self.match_data.loc[ind,'lose_reason'] = lose_reason
        return winner

    def __save_match(self, act_n):
        def __vector_transform( agent_index, vec):
            transformed_vec = vec
            if agent_index == 1:
                transformed_vec[0] = -transformed_vec[0]
                transformed_vec[1] = -transformed_vec[1]
            return transformed_vec

        def __coordinate_transform(agent_index, pos):
            mid_point = [177.5, 480]
            transformed_pos = pos-mid_point
            if agent_index == 0:
                transformed_pos[0] = -transformed_pos[0]
                transformed_pos[1] = -transformed_pos[1]
            return transformed_pos

        #print(self.match_data)
        world = self.world
        player_action = act_n[world.player]
        player = world.agents[world.player]
        opponent_action = act_n[1-world.player]
        opponent = world.agents[1-world.player]
        ball = world.ball
        ind = (world.match, world.rally, world.ball_round)

        #################################### state ####################################

        self.match_data.loc[ind, 'roundscore_player'] = world.score[world.player]
        self.match_data.loc[ind, 'roundscore_opponent'] = world.score[1-world.player]
        if world.player==0:
            self.match_data.loc[ind, 'player'] = 'A'
        else:
            self.match_data.loc[ind, 'player'] = 'B'
        #print(player_action['ball_type'])
        
        if ball.serve:
            self.match_data.loc[ind,'server'] = 1
        else:
            self.match_data.loc[ind,'server'] = 2
        ball_pos = __coordinate_transform(world.player, ball.state.p_pos.copy()) 
        self.match_data.loc[ind,'ball_location_x'] = ball_pos[0]
        self.match_data.loc[ind,'ball_location_y'] = ball_pos[1]
        self.match_data.loc[ind,'ball_height'] = ball.state.p_height
        ball_vel = __vector_transform(world.player, ball.state.p_vel.copy())
        self.match_data.loc[ind,'ball_velocity_x'] = ball_vel[0]
        self.match_data.loc[ind,'ball_velocity_y'] = ball_vel[1]
        self.match_data.loc[ind,'ball_velocity_z'] = ball_vel[2]

        player_pos = __coordinate_transform(world.player, player.state.p_pos.copy()) 
        player_vel = __vector_transform(world.player, player.state.p_vel.copy())
        self.match_data.loc[ind, 'playerA_location_x'] = player_pos[0]
        self.match_data.loc[ind, 'playerA_location_y'] = player_pos[1]
        self.match_data.loc[ind, 'playerA_velocity_x'] = player_vel[0]
        self.match_data.loc[ind, 'playerA_velocity_y'] = player_vel[1]
        self.match_data.loc[ind, 'playerA_ballround_distance'] = player.ball_round_distance

        oppo_pos = __coordinate_transform(world.player, opponent.state.p_pos.copy()) 
        oppo_vel = __vector_transform(world.player, opponent.state.p_vel.copy())
        self.match_data.loc[ind, 'playerB_location_x'] = oppo_pos[0]
        self.match_data.loc[ind, 'playerB_location_y'] = oppo_pos[1]
        self.match_data.loc[ind, 'playerB_velocity_x'] = oppo_vel[0]
        self.match_data.loc[ind, 'playerB_velocity_y'] = oppo_vel[1]
        self.match_data.loc[ind, 'playerB_ballround_distance'] = opponent.ball_round_distance

        self.match_data.loc[ind, 'time'] = self.time*self.world.time_scale
        ################################################################################

        #################################### action ####################################
        try:
            self.match_data.loc[ind, 'ball_type'] = ball.ball_type_table.loc[player_action['ball_type'],'chinese'] 
        except:
            self.match_data.loc[ind, 'ball_type'] = None
        
        self.match_data.loc[ind, 'launching_height'] = player_action['launching_height']
        if world.discrete_court == False:
            self.match_data.loc[ind, 'landing_location_x'] = player_action['landing_location'][0]
            self.match_data.loc[ind, 'landing_location_y'] = player_action['landing_location'][1]
            self.match_data.loc[ind, 'player_defend_x'] = player_action['player_location'][0]
            self.match_data.loc[ind, 'player_defend_y'] = player_action['player_location'][1]
            if world.decide_defend_location == True:
                self.match_data.loc[ind, 'opponent_defend_x'] = opponent_action['player_location'][0]
                self.match_data.loc[ind, 'opponent_defend_y'] = opponent_action['player_location'][1]
        else:
            self.match_data.loc[ind, 'landing_area'] = player_action['landing_area']
            self.match_data.loc[ind, 'player_defend_area'] = player_action['player_area']
            if world.decide_defend_location == True:
                self.match_data.loc[ind, 'opponent_defend_area'] = opponent_action['player_area']
        self.match_data.loc[ind, 'landing_height'] = player_action['landing_height']
        
        self.match_data.loc[ind, 'launching_height'] = player_action['launching_height']
        ################################################################################
        
        
    def __action_preprocessed(self, action_n):
        def __coordinate_transform(agent_index, pos):
            if None in pos:
                return np.array([None, None])
            mid_point = [177.5, 480]
            transformed_pos = pos
            
            if agent_index == 1:
                transformed_pos = -transformed_pos
            transformed_pos+= mid_point
            return transformed_pos
        player_action = action_n[ self.world.player]
        opponent_action = action_n[1-self.world.player]
        if self.world.discrete_court == True:
            player_action_requirement = ['landing_area', 'player_area', 'ball_type','landing_height']
            opponent_action_requirement = ['player_area']
        else:
            player_action_requirement = ['landing_location', 'player_location', 'ball_type']
            opponent_action_requirement = ['player_location']
        for key in player_action_requirement:
            assert key in player_action.keys(), \
                'Player {} action space error, required {}, got {}'.format(self.world.player, player_action_requirement,player_action.keys())
            #assert player_action[key] is not None,\
            #        'Player {} action[{}] = {} contain None'.format(self.world.player, key,player_action[key])
            if key in ['player_location', 'landing_location'] :
                #assert None not in player_action[key],\
                #    'Player {} action[{}] = {} contain None'.format(self.world.player, key,player_action[key])
                player_action[key] = __coordinate_transform(self.world.player,player_action[key]) 
        
        for key in opponent_action_requirement:
            assert key in opponent_action.keys(),\
                'Player {} action space error, required {}, got {}'.format(1-self.world.player, opponent_action_requirement,opponent_action.keys())
            #assert opponent_action[key] is not None,\
            #        'Player {} action[{}] = {} contain None'.format(1-self.world.player, key,opponent_action[key])
            if key in ['player_location', 'landing_location'] :
                #assert None not in opponent_action[key],\
                #    'Player {} action[{}] = {} contain None'.format(1-self.world.player, key,opponent_action[key])
                opponent_action[key] = __coordinate_transform(1-self.world.player,opponent_action[key]) 
        
        
                

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self,idx):
        def __vector_transform( agent_index, vec):
            transformed_vec = vec.copy()
            if agent_index == 1:
                transformed_vec[0] = -transformed_vec[0]
                transformed_vec[1] = -transformed_vec[1]
            return transformed_vec

        def __coordinate_transform(agent_index, pos):
            mid_point = [177.5, 480]
            transformed_pos = pos-mid_point
            if agent_index == 0:
                transformed_pos[0] = -transformed_pos[0]
                transformed_pos[1] = -transformed_pos[1]
            return transformed_pos

        ball =self.world.ball
        player = self.agents[idx]
        opponent = self.agents[1-idx]
        world = self.world

        match_obs = {
            'match':world.match,
            'rally':world.rally,
            'ball_round':world.ball_round,
            'score':world.score,
            'time':self.time*self.world.time_scale
        }
        if idx == 1:
            match_obs['score'] = match_obs['score'][::-1]
        ball_obs = {
            'player':world.player,
            'server':ball.serve,
            'location':__coordinate_transform(idx, ball.state.p_pos),
            'velocity':__vector_transform(idx, ball.state.p_vel),
            'height':ball.state.p_height
        }
        if ball.serve:
            ball_obs['serve'] = 1
        else:
            ball_obs['serve'] = 2
        player_obs = {
            'location':__coordinate_transform(idx,player.state.p_pos),
            'velocity':__vector_transform(idx,player.state.p_vel),
            'ball_round_distance':player.ball_round_distance
        }
        opponent_obs = {
            'location':__coordinate_transform(idx,opponent.state.p_pos),
            'velocity':__vector_transform(idx,opponent.state.p_vel),
            'ball_round_distance':opponent.ball_round_distance
        }
        
        
        return {
            'match':match_obs,
            'ball':ball_obs,
            'player':player_obs,
            'opponent':opponent_obs
        }

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, status):
        done = namedtuple(
            'done',('rally_done','match_done','all_match_done')
        )
        rally_done = None
        match_done = None
        all_match_done = None
        # 成功回擊
        if status == 1:
            rally_done = False
        else:
            rally_done = True

        score = self.world.score

        if score[0]==21 and score[1]<=19:
            match_done = True
        elif score[0]<=19 and score[1]==21:
            match_done = True
        elif score[0]>=20 and score[1]>=20:
            if abs(score[0]-score[1]) == 2:
                match_done = True
            else:
                match_done = False
        else:
            match_done = False
        if match_done:
            self.world.rally = 1
            self.world.match += 1
            self.world.score = [0, 0]
        
        if self.world.match > self.world.number_of_match:
            all_match_done = True
        
        return done(rally_done,match_done,all_match_done)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # reset rendering assets
    def _reset_render(self):
        self.top_view_render_geoms = None
        self.top_view_render_geoms_xform = None
        self.side_view_render_geoms = None
        self.side_view_render_geoms_xform = None

    # render environment
    def render(self, mode='human', info={}):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            #print(message)
        results = []
        if self.top_viewer:
            results.append(self.__top_view_render(mode, info)) 
 
            #print('render')
        #print(results)
        if  self.side_viewer and self.viewers[1] is None :
            #pass
            from multiagent import rendering
            #from g
            self.viewers[1] = rendering.Viewer(self.world.length,self.world.height)
            
        
        if self.side_view_render_geoms is None and self.side_viewer:
            from multiagent import rendering
            self.side_view_render_geoms = []
            self.side_view_render_geoms_xform = []
            for line in self.world.side_view_boundary:
                bound = []
                bound.append(line.start_coordinate )
                bound.append(line.end_coordinate )
                geom = rendering.make_polyline(bound)
                

                color = line.color
                geom.set_color(color[0],color[1],color[2],color[3])

                self.side_view_render_geoms.append(geom)
            
            for entity in self.world.entities:
                
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.side_view_render_geoms.append(geom)
                self.side_view_render_geoms_xform.append(xform)
            viewer = self.viewers[1]
            viewer.geoms = []
            for geom in self.side_view_render_geoms:
                viewer.add_geom(geom)
        if self.side_viewer:
            from multiagent import rendering
            viewer = self.viewers[1]
            viewer.set_bounds(0,self.world.length,0,self.world.height)
            for e, entity in enumerate(self.world.entities):
                
                
                if e ==2 :
                    scale = entity.state.p_pos[0]/150+1
                    z = entity.state.p_height
                else:
                    scale = entity.state.p_pos[0]/150
                    z = 0
                y = entity.state.p_pos[1]
                self.side_view_render_geoms_xform[e].set_scale(scale,scale)
                self.side_view_render_geoms_xform[e].set_translation(y, z)
            # render to display or array
            #print('render')
            results.append(viewer.render(return_rgb_array = mode=='rgb_array'))
        
        return results   

    def __top_view_render(self, mode='human', info={}):
        if  self.top_viewer and self.viewers[0] is None :
            # import rendering only if we need it (and don't import for headless machines)
            #
            from multiagent import rendering
            self.viewers[0] = rendering.Viewer(self.world.width,self.world.length)
        
        
       #print(self.render_geoms)
        # create rendering geometry
        results = []

        if self.top_view_render_geoms is None:
            #print('gg')
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.top_view_render_geoms = []
            self.top_view_render_geoms_xform = []

            for line in self.world.top_view_boundary:
                bound = []
                bound.append(line.start_coordinate )
                bound.append(line.end_coordinate )
                geom = rendering.make_polyline(bound)
                

                color = line.color
                geom.set_color(color[0],color[1],color[2],color[3])

                self.top_view_render_geoms.append(geom)
                
            for entity in self.world.entities:
                
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.top_view_render_geoms.append(geom)
                self.top_view_render_geoms_xform.append(xform)
            # add geoms to viewer
            
            viewer = self.viewers[0]
            viewer.geoms = []
            for geom in self.top_view_render_geoms:
                viewer.add_geom(geom)
            viewer.score = []
            for score in self.world.score:
                viewer.score.append(score)
            viewer.info = info
            info['ball_height'] = self.world.ball.state.p_height
            #print("123")

        
        if self.top_viewer:
            from multiagent import rendering
            viewer = self.viewers[0]
            viewer.set_bounds(0,self.world.width,0,self.world.length)
            for e, entity in enumerate(self.world.entities):
                self.top_view_render_geoms_xform[e].set_translation(*entity.state.p_pos)
                if e ==2 :
                    scale = self.world.ball.scale
                    self.top_view_render_geoms_xform[e].set_scale(scale,scale)
            # render to display or array
            
            result = viewer.render(self.world.score, return_rgb_array = mode=='rgb_array')
        return result
    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx

    

# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
