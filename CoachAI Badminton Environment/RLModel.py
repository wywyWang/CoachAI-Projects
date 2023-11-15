import pickle
from typing import Tuple, Literal
import numpy as np

"""
Inherited this class and overwrite action()
"""
class RLModel:
    def __init__(self):
        # finetund value that will add to original prob, 
        # first 3 dims represent the state, last dim is the prob
        # each prob of states should sum up to 0
        self.shot_finetuned = np.ones((10,10,10,11))
        self.move_finetuned = np.ones((10,10,10,10))
        self.land_finetuned = np.ones((10,10,10,10))

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

    """
    input:
        state: (player, opponent, ball)
            all is region no. from 1 ~ 10
    output:
        action: (shot, land, move)
            shot: 1~11, 11 is cannot reach
            land: region no. from 1 ~ 10, 10 is outside
            move: region no. from 1 ~ 10, 10 is outside
    """
    def action(self, state:Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]) \
        -> Tuple[Tuple[int, Tuple[float, float], Tuple[float, float]], Tuple[list, Tuple[list, list, list], list]]:
        raise NotImplementedError
    
    """
    input the action prob, and output its findtuned
    """
    def action_findtuned(self, state:Tuple[int,int,int], 
                         shot_prob:np.ndarray, land_prob:np.ndarray, move_prob:np.ndarray):
        state = state[0] - 1, state[1] - 1, state[2] - 1

        delta_shot = self.shot_finetuned[state]
        delta_move = self.move_finetuned[state]
        delta_land = self.land_finetuned[state]
        shot_prob *= delta_shot
        land_prob *= delta_land
        move_prob *= delta_move

        return (shot_prob/shot_prob.sum(),
                move_prob/move_prob.sum(),
                land_prob/land_prob.sum())

    def save(self, path:str):
        with open(path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
    