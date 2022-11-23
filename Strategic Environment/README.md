# RL-environment

## Introduction
This is a code of the paper **A Reinforcement Learning Badminton Environment for Simulating Player Tactics (Student Abstract)**. We aim to build a reinforcement learning environment that enable users to train, test agents and reproduce the match in a badminton game. Paper: https://arxiv.org/abs/2211.12234.

## Environment
The project is built with **Python 3.10** on **Visual Studio Code** on **Windows 10**.
You can directly download the project and run it on **Visual Studio Code** or you can download the code and include the following dependencies on your own environment.
## Dependency
```
numpy==1.22.3
pandas==1.4.2
scipy==1.8.0
sympy==1.10.1
pyglet==1.5.23
six==1.16.0
gym==0.23.1
```

You can install all dependencies above by **'pip -r requirment.txt'**. 

## Fold Structure
To make sure the evironment works, please check the code following the fold structure below.

```bash!
.
├── ...
├── 0816086                   
│   ├── main.py
│   ├── make_env.py
│   ├── Data
│   │   ├── ball_type_model.csv
│   │   ├── ball_type_table.xlsx
│   │   ├── hit_height.csv
│   │   ├── hit_range.csv
│   │   ├── player_speed_model.csv
│   │   └── player_speed.xlsx
│   ├── multiagent  
│   │   ├── scenarios
│   │   │   ├── __init__.py
│   │   │   └── badminton.py
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── environment.py
│   │   ├── policy.py
│   │   ├── rendering.py
│   │   └── scenario.py
└── ...

```

## Usage
To run the environment, please execute **main.py** and you should see the simulation window showing. If you want to modify the options in the environment, you can modify **main.py** or create your own starting file, follow the format in **Environment Introduction** and place it under the **0816086** folder.

## Environment Introduction
## Overview
![](https://i.imgur.com/W4ubCTF.png)
To modify the options of the environment, please change it as a parameter after declaring the class **world**.

### The important options in the environment


| parameter                | type    | defination                                                                                                                                                  | default |
| ------------------------ | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| discrete_court           | boolean | Whether to use the discretized court or not.</br> If so,  the court will be divided into 9 regions and the action related to the **x** and **y** coordinates will be ignored.</br> Otherwise, continuous court will be used, and action related to **area** will be ignored.              | False   |
| decide_defend_location   | boolean | Whether the players can decide the defensive location while the opponent is hitting.</br>If so, the player will go to the location given in action, and then go to the predicted landing location.</br>Otherwise, the player will go straight to the predicted landing location. | False   |
| number_of_match          | int     | The number of matches in a whole game.                                                                                                                                     | 3       |
| ball.returnable_height   | float   | The maximum height the player can return the ball. (scale: pixle)                                                                                                                  | 150     |
| ball.returnable_distance | float   | The maximum distance between player and ball that the player can return the ball. (scale: pixle)                                                                                      | float   | 150     |

## Objective coordinate system
Objective coordinate system is used by all players. You can assume that each player locate on the bottom court and observes the game via its own coordinate system.
<img src=https://i.imgur.com/4gmynaw.png, style='display: block;margin-left: auto;margin-right: auto;'></img>

The coordinate related parameters mentioned in **obs space** and **action space** below are using this coordinate system.



## Agent Observation
![](https://i.imgur.com/CkfrrnE.png)
**obs_n** includes two players' observations. **obs_n[0]** is the state for the player on the bottom court and **obs_n[1]** is for the top one. Each player will get a two-dimensional dictionary including 4 different kinds of states listed below.


### obs_n[n]
All the information related to coordinate in the state is using objective coordinate system.


| key        | type  | defination                |
| ---------- | ------ | ------------------------- |
| 'match'    | dict | The information about the match.  |
| 'ball'     | dict | The information about the current state of the ball.    |
| 'player'   | dict | The information about the agent himself. |
| 'opponent' | dict | The information about its opponent.      |
#### match


| key          | type     | defination             |
| ------------ | -------- | ---------------------- |
| 'ball_round' | int      | The number of ball rounds since current rally. |
| 'rally'      | int      | The number of rallies since the current match. |
| 'match'      | int      | The number of matches since the training started. |
| 'score'      | array(2) | The score of the current match.(P1:P2)      |
| 'time'       | float    | The time between now and the match starts. (sec.)                       |

#### ball


| key        | type     | defination                   |
| ---------- | -------- | ---------------------------- |
| 'location' | array(2) | The x and y coordinate of the shuttlecock when returning.(pixel)     |
| 'serve'           | boolean         | Whether the current ball round is the serving.         |
| 'height'   | float    | The height of the shuttlecock when returning.(pixel)           |
| 'velocity' | array(3) | The velocity in x, y, z direction before returning.(pixel) |
| 'player'   | int      |　Current player's id.                   |

#### player

| key        | type     | defination                     |
| ---------- | -------- | ------------------------------ |
| 'location' | array(2) | The x, y coordinate of himself.(pixel)     |
| 'velocity' | array(2) | The velocity of itself in x, y direction before returning.(pixel) |
| 'ball_round_distance' | float         |  The displacement of himself during the previous ball round.(pixel)                              |

#### opponent

| key        | value    | defination                     |
| ---------- | -------- | ------------------------------ |
| 'location' | array(2) | The x, y coordinate of the opponent.(pixel)     |
| 'velocity' | array(2) | The velocity of the opponent in x, y direction before returning.(pixel) |
| 'ball_round_distance' | float         |  The displacement of the opponent during the previous ball round.(pixel)                              |

## Action
The **act_n** includes the action of the agents from both sides. The **act_n[0]** represents the action of the player from the bottom court and the **act_n[1]** represents the top one.
:::warning
Notice that the coordinate system used in both players' actions is objective cooredinate system!
:::

### act_n[n]
**act_n[n]** is a dictionary.
#### When the agent is going to return the shuttlecock.(i.e. obs_n[n]['ball']['player']=n)
| key                | type     | defination                       | When to use                      |
| ------------------ | -------- | -------------------------------- | ------------------------------ |
| 'ball_type'        | int      | The style of current return.                   | Anytime                           |
| 'launching_height' | float    |The launching height of the serving. Only use when the agent is serving. | obs_n[n]['ball']['serve']=True |
| 'landing_location' | array(2) | The landing location of current return.           | world.discrete_court=False     |
| 'landing_area'     | int      | The id of the landing area of current return.       | world.discrete_court=True      |
| 'player_location'  | array(2) | The coordinate the player wants to go to after returning.     | world.discrete_court=False     |
| 'player_area'      | int      | The id of the area the player wants to go to after returning. | world.discrete_court=True      |
#### When the agent has just returned the shuttlecock and is allowed to decide the defensive location.(i.e. obs_n[n]['ball']['player']!=n and world.decide_defend_location=True)
| key               | type     | defination                       | When to use                  |
| ----------------- | -------- | -------------------------------- | -------------------------- |
| 'player_location' | array(2) | The coordinate the player wants to go to return the shuttlecock.     | world.discrete_court=False |
| 'player_area'     | int      | The id of the area the player wants to go to return the shuttlecock. | world.discrete_court=True  |




