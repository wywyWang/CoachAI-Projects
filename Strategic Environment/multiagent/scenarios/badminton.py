from numpy.lib.function_base import append
from multiagent.core import World, Agent, Boundary, Ball
from multiagent.rendering import make_polyline
from multiagent.scenario import BaseScenario
import numpy as np




class Scenario(BaseScenario):
    def make_world(self, resistance_force_n=1):
        world=World()
        world.resistance_force_n = resistance_force_n

        num_of_agent = 2
        num_of_adversary = 1

        world.agents = [Agent(i ,world.time_scale) for i in range(num_of_agent)]
        for i,agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.silent = True
            agent.adversary = True if i < num_of_adversary else False
            agent.size = 20

        ball = Ball(n=world.resistance_force_n, g=world.gravitational_acceration, time_scale=world.time_scale)
        ball.size = 10
        ball.movable = True
        world.ball = ball

        self.__draw_top_view(world)
        self.__draw_side_view(world)
    
        self.reset_world(world)

        return world

    def __draw_top_view(self, world):
        boundary = []
        boundary.append(Boundary([50,150],[50,810])) # left inner
        boundary.append(Boundary([30,150],[30,810])) # left outter
        boundary.append(Boundary([177.5,150],[177.5,810])) # mid
        boundary.append(Boundary([305,150],[305,810])) # right inner
        boundary.append(Boundary([325,150],[325,810])) # right outer
        
        boundary.append(Boundary([30,150],[325,150])) # bot outer
        boundary.append(Boundary([30,185],[325,185])) # bot inner
        boundary.append(Boundary([30,380],[325,380])) # mid top
        boundary.append(Boundary([30,480],[325,480])) # mid 
        boundary[-1].color = [1.0,0.0,0.0,1.0] # red
        boundary.append(Boundary([30,580],[325,580])) # mid bot
        boundary.append(Boundary([30,775],[325,775])) # top inner
        boundary.append(Boundary([30,810],[325,810])) # top outter

        world.top_view_boundary = boundary
    def __draw_side_view(self, world):
        boundary = []
        boundary.append(Boundary([480, 0],[480, 155/2]))
        world.side_view_boundary = boundary

    def reset_world(self, world):
        # Player 1
        world.agents[0].color = np.array([0.85, 0.35, 0.35])
        if world.score[world.player]%2==0:
            world.agents[0].state.p_pos = np.array([220.0,320.0])
        else:
            world.agents[0].state.p_pos = np.array([150,320.0])
        world.agents[0].state.p_vel = np.zeros(world.dim_p)
        world.agents[0].state.c = np.zeros(world.dim_c)

        # Player 2
        world.agents[1].color = np.array([0.35, 0.35, 0.85])
        if world.score[world.player]%2 ==0:
            world.agents[1].state.p_pos = np.array([150.0,660.0])
        else:
            world.agents[1].state.p_pos = np.array([220.0,660.0])
        world.agents[1].state.p_vel = np.zeros(world.dim_p)
        world.agents[1].state.c = np.zeros(world.dim_c)
        
        # Ball
        world.ball.color = np.array([0.0, 0.0, 0.0])
        world.ball.state.p_pos =np.array(world.agents[world.player].state.p_pos) 
        if world.player == 0:
            world.ball.state.p_pos[0]+=20
        else:
            world.ball.state.p_pos[0]-=20
        world.ball.state.p_vel = np.zeros(3)
        world.ball.is_flying = False
        
        world.ball.state.p_height = 100 
            
    def reward(self, agent, world):
        return 0
    def observation(self, agent, world):
        if not agent.adversary:
            return []
        else:
            return []

    

