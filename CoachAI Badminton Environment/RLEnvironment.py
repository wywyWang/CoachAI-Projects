import numpy as np
import os
from gym import spaces
from typing import Tuple, Literal

# shot type (1,2,3,4,5,6,7,8,9,10,11)
# 11 -> cannot reach
class Env:
    """
        state: (player, opponent, ball)
            all is region no. from 1 ~ 10
    """
    def __init__(self, max_iteration=1000):
        self.max_iteration = max_iteration
        self.state = (0., 0., 0., 0., 0., 0.)
        self.count = 0
        self.all = 0
        self.prev_player_coord = (0., 0.)
        self.shot_space = spaces.Box(0, 10, shape=(1,), dtype=np.int32)
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)   
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self._max_episode_steps = 1000

    def set_seed(self, seed: int = 10):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)

    # 客觀 -> 主觀
    """
    客觀座標轉為主觀座標
    客觀座標: (0, 0) 在場地左上角，以下為正
    主觀座標: (0, 0) 在場地中心

    player 1 在上半場， 2 在下半場
    """
    @staticmethod
    def obj2subj_coord(objective_coord: Tuple[float, float], player: Literal[1, 2]):
        x, y = objective_coord
        y = 960 - y  # move origin to left bottom
        if player == 2:
            x -= 177.5
            y -= 480
        elif player == 1:
            # rotate 180 deg
            x = 355 - x
            y = 960 - y
            x -= 177.5
            y -= 480
        else:
            NotImplementedError
        return x, y


    # 主觀 -> 客觀
    @staticmethod
    def subj2obj_coord(subjective_coord: Tuple[float, float], player: Literal[1, 2]):
        x, y = subjective_coord

        if player == 2:
            x += 177.5
            y += 480
        elif player == 1:
            x += 177.5
            y += 480

            # rotate 180 deg
            x = 355 - x
            y = 960 - y

        else:
            NotImplementedError
        y = 960 - y  # move origin to left bottom
        return x, y


    def reset(self):
        # self.state = ((0., 0.), (0., 0.), (0., 0.))
        # self.state = (0., 0., 0., 0., 0., 0.)
        player_x = np.random.uniform(0, 127.5)
        if np.random.random() < 0.5:
            player_x = player_x * -1
        player_y = np.random.uniform(-330, 0)

        opponent_x = np.random.uniform(-127.5, 0)
        if np.random.random() < 0.5:
            opponent_x = opponent_x * -1
        opponent_y = np.random.uniform(0, 330)

        ball_x = np.random.uniform(-127.5, 0)
        if np.random.random() < 0.5:
            ball_x = ball_x * -1
        ball_y = np.random.uniform(0, 330)
        
        self.state = [player_x, player_y, opponent_x, opponent_y, ball_x, ball_y]

        # opp y: 114 276 x -127.5 0
        # play y: -114 -276 x 0 127.5
        return self.state
    
    """
    input:
        action: (shot, land, move)
            shot: 1~11, 11 is cannot reach
            land: region (x, y)
            move: region (x, y)
    output:
        (nextstate, reward)
    """
    def step(self, action:Tuple[int, Tuple[float, float], Tuple[float, float]], is_launch):
        shot_type, land_coord, move_coord = action
        # landing_x, landing_y= land_coord
        # move_x, move_y = move_coord

        landing_x, landing_y= land_coord
        move_x, move_y = move_coord
             
        next_state_player = self.prev_player_coord
        next_state_opponent = -move_coord[0], -move_coord[1]
        next_state_ball = -land_coord[0], -land_coord[1]

        self.state = (next_state_player, next_state_opponent, next_state_ball)
        if not is_launch and np.sqrt((self.prev_player_coord[0] + landing_x)**2 + (self.prev_player_coord[1] + landing_y)**2) > (141.32845494778113 + 58.18920927215389):
            return self.state, -1

        if not is_launch and move_y > 0:
            return self.state, -1

        if is_launch:
            if shot_type in [2,3,4,5,6,7,8,9,11]: # not launch ball
                return self.state, -1
        else:
            if shot_type in [1, 10, 11]:  # launch ball
                return self.state, -1
        
        if landing_x < -127.5 or landing_x > 127.5 or landing_y < 0 or landing_y > 330: #outside
            return self.state, -1
        # rotate 180 deg
        self.prev_player_coord = move_coord
        
        return self.state, 0.04