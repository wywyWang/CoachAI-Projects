#import dis
#from html import entities
#from importlib.metadata import distribution
#from sqlite3.dbapi2 import _AnyParamWindowAggregateClass
#from turtle import distance
#from xml.etree.ElementTree import TreeBuilder

import numpy as np
import pandas as pd
import scipy.stats as stats
import pickle
import os

from sympy import false



# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

        

        
        self.ball_type_word_to_num = {
    '挑球': 0, '推球': 1, '放小球': 2, '擋小球': 3, '平球': 4, '小平球': 5,
    '撲球': 6, '防守回挑': 7,  '勾球': 8,  '後場抽平球': 9,
    '點扣': 10, '長球': 11,  '殺球': 12, '防守回抽': 13,
    '過度切球': 14, '切球': 15, '發短球': 16,  '發長球': 17,
    '未知球種': 18
}

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self, index, time_scale):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        
        self.agent_index = index

        self.time_scale = time_scale
        self.ball_round_distance = 0
    
        self.height = 180/2
        self.racket_length = 70/2
        try:
            self.palyer_speed_model = pd.read_csv(r'0816086\Data\player_speed_model.csv',index_col=0)
        except:
            self.palyer_speed_model = pd.read_csv(r'rl-environment\0816086\Data\player_speed_model.csv',index_col=0)
        self.max_speed = 10/2*100

        self.move_length = []
    def step(self):
        #print('Agent',self.agent_index,'target pos',self.target_pos, 'current pos',self.state.p_pos, 'vel',self.state.p_vel)
        if (self.target_pos == self.state.p_pos).all():
            self.state.p_vel = np.zeros(2)
            return
        
        distance_to_target_pos = np.linalg.norm(self.state.p_pos-self.target_pos)
        

        if distance_to_target_pos <= np.linalg.norm(self.state.p_vel):
            self.state.p_pos = self.target_pos
            self.ball_round_distance += distance_to_target_pos
            if self.role == 'opponent' :
                
                if self.__in_self_court(self.ball.target_pos) and self.chase_ball:
                    self.chase_ball = False
                    target_pos = self.ball.ideal_pos.copy()
                    #print('target_pos',target_pos)
                    self.generate_speed(target_pos)
                    
                    #print('chase ball')
                
        else:
            self.ball_round_distance += np.linalg.norm(self.state.p_vel)
            self.state.p_pos = self.state.p_pos + self.state.p_vel
        #print('pos',self.state.p_pos)
        
    
    def initialize_step(self, role, decide_defend_location , agent_target_pos=None, ball=None):
        self.role = role
        self.ball_round_distance = 0
        if role=='player':
            target_pos = agent_target_pos
        
        if role == 'opponent' and decide_defend_location==False:
            if self.__in_self_court(ball.target_pos):
                #print('ball_ideal_pos',ball.ideal_pos)
                target_pos = ball.ideal_pos.copy()
            else:
                target_pos = self.state.p_pos
        #print('role',role)
        #print('ball targt',ball.target_pos)
        #print('self.state.pos',self.state.p_pos)
        #print('target',target_pos)
        self.ball = ball
        self.chase_ball = True
        self.generate_speed(target_pos)  
        

    def generate_speed(self, target_pos):
        role = self.role
        time_scale = self.time_scale
        if (target_pos==None).any():
            self.state.p_vel = np.zeros(2)
            return
        if np.isnan(target_pos).any():
            self.state.p_vel = np.zeros(2)
            return
       

        self.target_pos = target_pos
        
        
        distance = np.linalg.norm(target_pos-self.state.p_pos)
        #print(self.agent_index,' distance',distance)
        
        intercept = self.palyer_speed_model.loc[role, 'intercepts']
        fac1 = self.palyer_speed_model.loc[role, 'fac1']
        fac2 = self.palyer_speed_model.loc[role, 'fac2']
        degree = self.palyer_speed_model.loc[role, 'degree']
        self.state.p_vel = intercept 
        if degree!=1:
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=2, include_bias=False)
            features = poly.fit_transform([[distance]])[0]
            self.state.p_vel+= (fac1*features[0]+fac2*features[1])
            
        else:
            self.state.p_vel+=fac1*distance      
       # print('player velocity',self.p_vel) 
        if self.state.p_vel < 0:
            self.state.p_vel = 0
        if self.state.p_vel > self.max_speed:
            self.state.p_vel = self.max_speed
        if (self.target_pos != self.state.p_pos).any():
            direction_vec = self.target_pos - self.state.p_pos
            unit_direction_vec = direction_vec/np.linalg.norm(direction_vec)
            self.state.p_vel = time_scale*unit_direction_vec*self.state.p_vel
        else:
            self.state.p_vel = np.zeros(2)
       # print('agent p_vel:',self.state.p_vel) 

    def __in_self_court(self, pos):
        if pos[1] < 150 or pos[1] > 810 or pos[0] < 50 or pos[0] >305:
            return False
        
        if self.agent_index == 0:
            if pos[1]<=480:
                return True
        elif pos[1] >= 480:
            return True
        return False

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 2
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # size of court
        self.length = 960
        self.width = 355
        self.height = 600
        # height of net
        self.net_height = 155/2
        # physical parameter
        self.gravitational_acceration = 9.8
        self.resistance_force_n = 1

        self.init_step = True

        self.discrete_court = False
        self.decide_defend_location = False
        self.number_of_match = 3

        self.t = 0.0
        self.time_scale = 1.0/60

       
        self.player = 1

        # match counter
        self.rally = 1
        self.ball_round = 1
        self.match = 1
        self.score = [0,0]
    # return all entities in the world
    @property
    def entities(self):
        if self.ball:
            return self.agents + self.landmarks + [self.ball]
        return self.agents + self.landmarks 

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    
    
    def area_mapping(self,agent_index,area):
        
        if area in np.array([2,4,6]):
            target_x = (50+104)/2
        elif area in np.array([7,8,9]):
            target_x = 177.5
        else:
            target_x = (305+177.5)/2

        if area in np.array([1,2,7]):
            target_y = 480-114/2
        elif area in np.array([5,6,8]):
            target_y = 480-114-162/2
        else:
            target_y = 480-114-162-54/2

        if agent_index == 1:
            target_x = target_x+2*(177.5-target_x)
            target_y = 480+abs(480-target_y)
        return np.array([target_x,target_y])
    
    def pre_step(self, action_n):
        
        ball = self.ball
        launcher = self.entities[self.player]
        launcher_action = action_n[self.player]
        catcher = self.entities[(1-self.player)]
        ball.ball_type = launcher_action['ball_type']

        
        
        if ball.serve:
            ball.returned = True
            return False
        
        ball.step()
        catcher.step()
        launcher.step()
        
        self.t += 1

        finish, _ = self.check_step_done(ball, launcher)
        if finish:
            ball.returned = True
            return False
        return True

    # update state of the world
    def step(self,action_n):
        ball = self.ball

        launcher = self.entities[self.player]
        catcher = self.entities[(1-self.player)]
        launcher_action = action_n[self.player]
        catcher_action = action_n[(1-self.player)]
        ball.ball_type = launcher_action['ball_type']
        
        
        if self.init_step:
            if self.discrete_court:
                ball_target_area = launcher_action['landing_area']
                landing_pos = self.area_mapping(1 if self.player==0 else 0,ball_target_area)
            else:
                landing_pos = launcher_action['landing_location']
            
            
            #print(landing_pos)
            ball_type = launcher_action['ball_type']
            landing_height = launcher_action['landing_height']
            if ball.serve:
                if ball_type != 16 and ball_type != 17:
                    #發球失敗
                    return  4
                if landing_height is None or np.isnan(landing_height):
                    #預設高度
                    landing_height = 0
                launching_height = launcher_action['launching_height']
                ball.launching(ball_type, landing_pos, landing_height, launching_height)
            else:
                #不回擊
                if ball_type is None:
                    pass
                #回擊失敗
                elif ball_type<0 or ball_type>17:
                    pass
                #正常回擊
                else:
                    if landing_height is None or np.isnan(landing_height):
                    #預設高度
                        landing_height = 0
                    ball.launching(ball_type, landing_pos, landing_height)

            for ind, act in enumerate(action_n):
                agent = self.agents[ind]
                action_n[ind] = act
                #print('player',self.player)
                if agent == launcher:
                    role = 'player'
                else:
                    role = 'opponent'
                if self.discrete_court:
                    agent_target_area = act['player_area']
                    agent_target_pos = self.area_mapping(ind, agent_target_area)
                else:

                    agent_target_pos = act['player_location']
                #print(agent_target_pos)
                agent.initialize_step(role, self.decide_defend_location, agent_target_pos, ball)
            
            self.t = 0.0
            self.init_step = False
        
        
        

        
        step_done, status = self.check_step_done(ball, catcher)
       # print('status:',status)
        if step_done:
            self.init_step = 1
            self.falll_into_region = False
            ball.serve = False
            ball.returned = False
            return status
        else:
            ball.step()
            launcher.step()
            catcher.step()
        
            self.t += 1
            return status

    def check_step_done(self, ball, catcher):
        distance_between_ball_and_catcher = np.linalg.norm(catcher.state.p_pos-ball.state.p_pos)
        # Check return 
        if ball.returnable(distance_between_ball_and_catcher):
            #print('distance between catcher and ball:',distance)
            #print('ball height:', ball.state.p_height)
            return True, 1
        # Check passed net
        if ball.passed_net == False:
            # From our side
            #print(self.player)
            #print(ball.state.p_pos)
            if self.player == 1 and ball.state.p_pos[1]<=480:
                if ball.state.p_height >= self.net_height:
                    ball.passed_net = True
                else:
                    ball.state.p_pos[1] = 480+5
                    ball.state.p_height = 0
                    return True, 3
            if self.player == 0 and ball.state.p_pos[1]>=480:
                if ball.state.p_height >= self.net_height:
                    ball.passed_net = True
                    
                else:
                    ball.state.p_pos[1] = 480-5
                    ball.state.p_height = 0
                    return True, 3
            

        # Check drop
        if ball.state.p_height<=0:
            ball.state.p_height = 0
            return True, 2
        return False, 0

class Boundary(Entity):
    def __init__(self):
        super(Boundary, self).__init__()

        self.start_coordinate = [0,0]
        self.end_coordinate = [0,0]

        self.color = [0,0,0,1]

    def __init__(self,start,end):
        super().__init__()

        self.start_coordinate = start
        self.end_coordinate = end

        self.color = [0,0,0,1]

class Ball(Entity):
    def __init__(self, n, g, time_scale):
        super(Ball, self).__init__()
        
        self.g = g
        self.n = n
        self.m = 5.19/1000 #(kg)
        #1.羽球終端速度範圍為 5.84 m/s ~ 6.48 m/s
        self.V_T = 6.86
        self.b = (self.m*g) / (self.V_T**n)
        try:
            self.ball_type_table = pd.read_excel(r'Data\ball_type_table.xlsx')
            self.ball_initial_state_model = pd.read_csv(r'Data\initial_state\ball_initial_state_model.csv',index_col=0)
            self.ball_type_frame_distribution = pd.read_csv(r'Data\ball_type_model.csv',index_col=0)
            self.hit_height = pd.read_csv(r'Data\hit_height.csv',index_col='type')
            self.hit_range = pd.read_csv(r'\Data\hit_range.csv',index_col='type')
    
        except:
            self.ball_type_table = pd.read_excel(r'0816086\Data\ball_type_table.xlsx')
            self.ball_initial_state_model = pd.read_csv(r'0816086\Data\initial_state\ball_initial_state_model.csv',index_col=0)
            self.ball_type_frame_distribution = pd.read_csv(r'0816086\Data\ball_type_model.csv',index_col=0)
            self.hit_height = pd.read_csv(r'0816086\Data\hit_height.csv',index_col='type')
            self.hit_range = pd.read_csv(r'0816086\Data\hit_range.csv',index_col='type')
        
        self.state.p_vel = 0
        self.state.p_height = 0

        self.returnable_distance = 150
        self.returnable_height = 150
        self.serve = True

        self.returned = False
        self.t= 0.
        self.time_scale = time_scale

    def launching(self, ball_type, landing_pos, landing_height, init_height=None):
        #assert ball_type is not None, 'Fail to return because ball type is None'
        #assert ball_type>=0 and ball_type<=17, 'Fail to return because ball type {} is invalid'.format(ball_type)
        assert not (landing_pos==None).any(), 'Fail to return bacause landing position {} is invalid'.format(landing_pos)
        assert landing_height is not None, 'Fail to return bacause landing height {} is invalid'.format(landing_height)
        assert  not np.isnan(landing_height), 'Fail to return bacause landing height {} is invalid'.format(landing_height)

        if init_height is None:
            self.initial_height = self.state.p_height
        else:
            self.initial_height = init_height

        self.target_pos = landing_pos
        distance = np.linalg.norm(self.state.p_pos-self.target_pos)
       # print(landing_height)
        #print(ball_type, distance, landing_height-self.state.p_height)
        speed, angle_arc = self.__generate_deterministic_speed_and_angle(ball_type, distance, landing_height-self.state.p_height)
        angle_degree = angle_arc*360/2/np.pi

        #print('ball initial speed(m/s):',speed)
        #print('ball launching angle(degree):',angle_degree)

        
        self.t = 0
        self.launching_angle = angle_arc
        self.initial_speed = speed
        self.initial_pos = self.state.p_pos.copy()
        self.passed_net = False
        self.serve = False

    def step(self):
        self.t += self.time_scale
        t= self.t
        
        self.state.p_pos, self.state.p_height = self.__get_position(t)
        

        diff = 0.0001
        tmp_pos, tmp_height = self.__get_position(t+diff)
        horizontal_vel = (tmp_pos-self.state.p_pos)/diff
        verticle_vel = (tmp_height-self.state.p_height)/diff
        self.state.p_vel = [*horizontal_vel, verticle_vel]
        #print(self.state.p_vel)

        
    def __get_position(self, t):
        x = self.__get_position_x(t)
        y = self.__get_position_y(t)
        pos = np.array([x,y])

        direction_vec = self.target_pos - self.initial_pos
        unit_direction_vec = direction_vec/np.linalg.norm(direction_vec)
        move_vec = unit_direction_vec*pos[0]

        p_pos = self.initial_pos + move_vec
        height = self.initial_height + pos[1]
        
        return p_pos, height
    
    @property
    def ideal_pos(self):
        t = self.t
        if self.returned == True or self.ball_type is None:
            ideal_height = 220/2
        else:
            type_eng = self.ball_type_table.loc[self.ball_type, 'english']
            ideal_height = self.hit_height.loc[type_eng, 'mean']
        #print('ideal_height',ideal_height)
        #print('current height',self.state.p_height)
        ideal_height = min(ideal_height, self.state.p_height) 
        while True:
            t += 1.0
            p_pos, height = self.__get_position(t)
            height = self.initial_height+height
            #print(height)
            if height<=0:
                break
        while True:
            t -= 0.1
            p_pos, height = self.__get_position(t)
            height = self.initial_height+height
            if height>=ideal_height or t <= self.t+1:
                break
        #print(flying_time)
        #print('ideal_pos',p_pos)
        return p_pos


    def returnable(self, distance):
        #print(distance)
        # Too far from player
        if distance>self.returnable_distance:
            #print('too far')
            return False
        # Haven't passed the net
        if self.passed_net == False:
            #print('havent passed net')
            return False
        # The ball is flying and the player haven't make decision
        if self.returned == True:
            if self.state.p_height <= self.returnable_height:
               return True
            #else:
              #  print('havent returned too high')
        # The ball is closed to player, return only rational height
        elif self.ball_type is not None:
            
            type_eng = self.ball_type_table.loc[self.ball_type, 'english']
            mean = self.hit_height.loc[type_eng, 'mean']
            std = self.hit_height.loc[type_eng, 'std']
            #print('mean',mean,'std',std)
            if self.state.p_height <= mean+3*std and self.state.p_height >= mean-3*std:
                mean = self.hit_range.loc[type_eng, 'mean']
                std = self.hit_range.loc[type_eng, 'std']
                
                if distance <= mean+3*std and distance >= mean-3*std:
                    return True
                else:
                   # print('have returned but too far')
                   # print('current distance:',distance)
                   # print('accept distance:',mean-2*std,mean+2*std)
                    return False
            else:
                pass
                #print('have returned but too high or too low')
                #print('current height:',self.state.p_height)
                #print('accept height:',mean-2*std,mean+2*std)
           
        return False

    def __get_position_x(self, t):
        g = self.g
        n = self.n
        V_T = self.V_T
        V_xi = self.initial_speed * np.cos(self.launching_angle)
        if n == 1:
            x =  (V_T*V_xi)/g * (1-np.exp(-g*t/V_T))
        elif n == 2:
            x = V_T**2/g*np.log((V_xi*g*t+V_T**2)/V_T**2) # m

        return x*100/2 # pixel
    
    def __get_position_y(self, t):
        g = self.g
        n = self.n
        m = self.m
        V_T = self.V_T
        V_yi = self.initial_speed * np.sin(self.launching_angle)
       
        if n==1:
            y = V_T/g*(V_T+V_yi)*(1-np.exp(-g*t/V_T))-V_T*t
        elif n==2:
            numerator = np.sin(g*t/V_T+np.arctan(V_T/V_yi))
            denominator = np.sin(np.arctan(V_T/V_yi))
            y = V_T**2/g*np.log(abs(numerator/denominator))
        return y*100/2

    def __generate_stochastic_speed_and_angle(self, ball_type, distance):
        
            
        table = self.ball_initial_state_model
        ball_type = self.ball_type_table.loc[ball_type,'english']
        #print(table)
        #print(ball_type)
        speed_slope = table.loc[ball_type, 'speed_slope']
        speed_intercept = table.loc[ball_type, 'speed_intercepts']
        #print(speed_slope)
        #print(speed_intercept)
        #print(distance)
        speed = distance*speed_slope + speed_intercept

        angle_mean = table.loc[ball_type, 'angle_mean']
        angle_std = table.loc[ball_type, 'angle_std']
        #print(table.loc[ball_type, 'angle_mean'])
        #print( table.loc[ball_type, 'angle_std'])
        #print(np.random.normal(angle_mean, angle_std, 1))
        angle = np.random.normal(angle_mean, angle_std/2, 1)[0]

        return float(speed), float(angle)

    def __generate_deterministic_speed_and_angle(self, ball_type, distance, height):
        #print(self.ball_type_frame_distribution)
        table = self.ball_initial_state_model
        ball_type = self.ball_type_table.loc[ball_type,'english']
        #angle_mean = table.loc[ball_type, 'angle_mean']
        
        mean = self.ball_type_frame_distribution.loc[ball_type,'mean']
        std = self.ball_type_frame_distribution.loc[ball_type,'std']
        

        frame = np.random.normal(mean, 0.00001, 1)[0]

        def guess_initial(Ve,g,x_pixel,y_pixel,t):
            
            Ve = Ve   #end velovity (m/s)
            g = g    #gravity acc (m/s**2)
            x = x_pixel*2/100    #xy dif (pixel*2/100 = m)
            y = y_pixel*2/100   #height dif (pixel*2/100 = m)
            t = t/30   #time dif (frame/30 = s)

            if self.n == 1:
                Vxi = x*g/(1-np.exp(-g*t/Ve))/Ve
                Vyi = (y+Ve*t)/(1-np.exp(-g*t/Ve))*g/Ve-Ve
            if self.n == 2:
                
                Vyi = Ve*(np.exp(g*y/Ve**2)-np.cos(g*t/Ve))/np.sin(g*t/Ve)
                Vxi = (Ve**2*np.expm1(g*x/Ve**2))/(g*t)

            

            Vi = np.sqrt(((Vxi+Vyi)**2+(Vxi-Vyi)**2)/2)
            a = np.arcsin(Vyi/Vi)
            ans = [Vi, a]


            return ans #[v,a]
        
        def guess_initial2(Ve, g, x_pixel, y_pixel, angle):
            Ve = Ve   #end velovity (m/s)
            g = g    #gravity acc (m/s**2)
            x = x_pixel*2/100    #xy dif (pixel*2/100 = m)
            y = y_pixel*2/100   #height dif (pixel*2/100 = m)
            a = angle
            Vi = symbols('Vi')
            
            f1 = Function('f')

            f1 = x*(Ve+Vi*sy.sin(a))/(Vi*sy.cos(a))-Ve**2/g*sy.log(Ve*Vi*sy.cos(a)/(Ve*Vi*sy.cos(a)-g*x))-y
            
            S2 = solve([f1],[Vi])
            ans = [S2[0], a]
            return ans
            #Vi = np.sqrt(((Vxi+Vyi)**2+(Vxi-Vyi)**2)/2)
            
        #sprint(distance)
        #
        print(distance, height, frame)
        ans = guess_initial(self.V_T, self.g, distance, height, frame)
        
        return float(ans[0]), float(ans[1])

    @property
    def scale(self):
        return self.state.p_height / 150 + 1
