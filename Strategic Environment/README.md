# A Reinforcement Learning Badminton Environment for Simulating Player Tactics (AAAI 2023 Student Abstract)
Official code of the paper **A Reinforcement Learning Badminton Environment for Simulating Player Tactics**.
Paper link will be released in the near future.

## Introduction
Recent techniques for analyzing sports precisely has stimulated various approaches to improve player performance and fan engagement. However, existing approaches are only able to evaluate offline performance since testing in real-time matches requires exhaustive costs and cannot be replicated. To test in a safe and reproducible simulator, we focus on turn-based sports and introduce a badminton environment by simulating rallies with different angles of view and designing the states, actions, and training procedures. This benefits not only coaches and players by simulating past matches for tactic investigation, but also researchers from rapidly evaluating their novel algorithms.

## 環境
![](https://i.imgur.com/W4ubCTF.png)
與環境相關的重要參數請在宣告**world**後，作為**world**物件變數設定。
### 環境重要參數


| parameter                | type    | defination                                                                                                                                                  | default |
| ------------------------ | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| discrete_court           | boolean | 是否使用離散型的場地。</br>若是，則將場地劃分為9區，與x、y座標相關的action將被忽略；</br>若否，則使用連續型場地，與area相關的action將被忽略。               | False   |
| decide_defend_location   | boolean | 防守的球員在對方擊球的同時是否能決定接球的位置。若是，則球員優先前往action指定的接球位，再前往預測的球落點位置；</br>若否，則球員直接前往預測的球落點位置。 | False   |
| number_of_match          | int     | 一場完整的比賽要有幾局                                                                                                                                      | 3       |
|    ball.returnable_height                    |  float       |  球的高度小於多少進入可回擊範圍，單位:pixle                                                                                                                                                           |   150      |
| ball.returnable_distance | float   | 球與球員距離小於多少時進入可回擊範圍，單位為:pixle                                                                                                            | 150     |

## 主觀座標系
所有球員皆使用自身的主觀坐標系，皆假定自己在下半場而對手在上半場，座標系如下圖所示:
<img src=https://i.imgur.com/4gmynaw.png, style='display: block;margin-left: auto;margin-right: auto;'></img>
以下 obs space 與 action space 所提及之座標相關參數皆使用此座標系。

## Agent Observation
![](https://i.imgur.com/CkfrrnE.png)
**obs_n** 包含兩個 agent 的 observation，其中**obs_n[0]** 為下半場球員的 state，**obs_n[1]** 則為上半場球員的 state。
每個球員獲得的 state 為一個二維 dictionary，其中包含下列4種不同的類型的 state。
### obs_n[n]
state中所有座標相關資訊都經過座標轉換，為agent的主觀座標系，詳情請見主觀座標系定義。

| key        | type  | defination                |
| ---------- | ------ | ------------------------- |
| 'match'    | dict | 包含與比賽本身相關的資訊  |
| 'ball'     | dict | 包含與當顆球相關的資訊    |
| 'player'   | dict | 包含與agent本身相關的資訊 |
| 'opponent' | dict | 包含與對手相關的資訊      |
#### match


| key          | type     | defination             |
| ------------ | -------- | ---------------------- |
| 'ball_round' | int      | 當拍數是當回合的第幾拍 |
| 'rally'      | int      | 當回合是當局的地幾回合 |
| 'match'      | int      | 當局從訓練開始的第幾局 |
| 'score'      | array(2) | 當局的比分(P1:P2)      |
| 'time'       | float    | 從比賽開始到回擊當下過了幾秒                       |

#### ball


| key        | type     | defination                   |
| ---------- | -------- | ---------------------------- |
| 'location' | array(2) | 球在回擊當下的 x、y 座標     |
| 'serve'           | boolean         | 本球是否為發球                             |
| 'height'   | float    | 球在回擊當下的高度           |
| 'velocity' | array(3) | 球在被回擊前 x、y、z 方向的速度 |
| 'player'   | int      | 擊球選手id                   |

#### player

| key        | type     | defination                     |
| ---------- | -------- | ------------------------------ |
| 'location' | array(2) | 我方在回擊當下的 x、y 座標     |
| 'velocity' | array(2) | 我方在被回擊前 x、y 方向的速度 |
| 'ball_round_distance' | float         |  在前一個ball_round移動的距離(pixel)                              |

#### opponent

| key        | value    | defination                     |
| ---------- | -------- | ------------------------------ |
| 'location' | array(2) | 對方在回擊當下的 x、y 座標     |
| 'velocity' | array(2) | 對方方在被回擊前 x、y 方向的速度 |
| 'ball_round_distance' | float         |  在前一個ball_round移動的距離(pixel)                              |

## Action
**act_n** 包含兩個 agent 的 action，其中**act_n[0]** 為下半場球員的 action，**act_n[1]** 則為上半場球員的 action。
請注意雙方球員的action使用的座標系皆為主觀座標系!
### act_n[n]
**act_n[n]** 本身為一個dict。
#### 當球員為進攻方時(i.e. obs_n[n]['ball']['player']=n)
| key                | type     | defination                       | 使用時機                       |
| ------------------ | -------- | -------------------------------- | ------------------------------ |
| 'ball_type'        | int      | 本球回擊的球種                   | 隨時                           |
| 'launching_height' | float    | 發球時的擊球高度(只在發球時使用) | obs_n[n]['ball']['serve']=True |
| 'landing_location' | array(2) | 本球回擊的目標落點座標           | world.discrete_court=False     |
| 'landing_area'     | int      | 本球回擊的目標落點區域代號       | world.discrete_court=True      |
| 'player_location'  | array(2) | 我方球員擊球後移動的目標座標     | world.discrete_court=False     |
| 'player_area'      | int      | 我方球員擊球後移動的目標區域代號 | world.discrete_court=True      |
#### 當球員為防守方且允許防守方決定防守位置(i.e. obs_n[n]['ball']['player']!=n and world.decide_defend_location=True)
| key               | type     | defination                       | 使用時機                   |
| ----------------- | -------- | -------------------------------- | -------------------------- |
| 'player_location' | array(2) | 我方球員防守的目標座標     | world.discrete_court=False |
| 'player_area'     | int      | 我方球員防守的目標區域代號 | world.discrete_court=True  |