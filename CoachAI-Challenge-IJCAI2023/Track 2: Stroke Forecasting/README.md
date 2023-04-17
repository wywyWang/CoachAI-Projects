# Track 2: Forecasting Future Turn-Based Strokes in Badminton Rallies

## :badminton: Task Introduction
The goal of this track is to **forecast future strokes including shot types and locations given the past stroke sequences**, namely stroke forecasting. For each singles rally, given the observed 4 strokes with type-area pairs and two players, the goal is to predict the future strokes including shot types and area coordinates for the next n steps. n is various based on the length of the rally.

## :badminton:	Data Overview
* Input: landing_x, landing_y, shot type of past 4 strokes 
* Output: landing_x, landing_y, shot type of future strokes 
### Column Meaning
* train.csv, val.csv, test.csv
  * rally: serial number of rallies in a match
  * ball_round: the order of the shot in a rally
  * time (hr:min:sec): the shot’s hitting time
  * frame_num: sec * fps = frame_num
  * roundscore_A: Player A’s current score in the set
  * roundscore_B: Player B’s current score in the set
  * player: the player who performed the shot
  * type: the type of shot, i.e., short service, long service, clear, push/rush, smash, defensive shot, drive, net shot, lob, drop. (the naming might be different for the two tracks)
  * aroundhead: hit the shuttle around the head or not
  * backhand: hit the shuttle with back hand or not
  * landing_height: if the shuttle destinations is hit above (1) or below (0) the net
  * landing_area: the grid of the shuttle destinations
  * landing_x, landing_y: the coordinates of the shuttle destinations
  * lose_reason: the reason why the rally ended. To be capable of different usages, we record the timest
  * getpoint_player: the player who won the rally
  * player_location_area: the location of the player who performed the shot
  * player_location_x, player_location_y: the coordinates of the player’s location when the player hits the shuttle
  * opponent_location_area: the location of the player who prepared to receive the shot
  * opponent_location_x, opponent_location_y: the coordinates of the opponent’s location when the player hits the shuttle
  * set: the current set in a match (best-of-3)
  * match_id: serial number of matches
  * rally_id: serial number of all rallies
  * rally_length: number of shots in a rally, rally_length-4 is the shots needed to predict
* match_metadata.csv
  * match_id: serial number of matches
  * set: the number of played sets in a match (either 2 or 3)
  * duration: the duration of a match in minutes
  * winner: the winner player
  * loser: the lose player
  * homography_matrix: Help transform coordinates from the real-world system back to the camera system by p=H-1p’
* For getting access to the data, please fill in the [form](https://forms.gle/znfgo4Bvp3t9h8wk9).


## :badminton:	Problem Definition

Let $R=\{S_r, P_r\}_{r=1}^{|R|}$ denote historical rallies of badminton matches, where the $r$-th rally is composed of a stroke sequence with type-area pairs $S_r=(\langle s_1, a_1\rangle,\cdots,\langle s_{|S_r|}, a_{|S_r|}\rangle)$ and a player sequence $P_r=(p_1,\cdots,p_{|S_r|})$.
At the $i$-th stroke, $s_i$ represents the shot type, $a_i=\langle x_i, y_i\rangle \in \mathbb{R}^{2}$ are the coordinates of the shuttle destinations, and $p_i$ is the player who hits the shuttle. We denote Player A as the served player and Player B as the other for each rally in track2. For instance, given a singles rally between Player A and Player B, $P_r$ may become $(A, B, \cdots, A, B)$.
We formulate the problem of stroke forecasting as follows. For each rally, given the observed $\tau$ strokes $(\langle s_i, a_i\rangle)_{i=1}^{\tau}$ with players $(p_i)_{i=1}^{\tau}$, the goal is to predict the future strokes including shot types and area coordinates for the next $n$ steps, i.e., $(\langle s_i, a_i\rangle)_{i={\tau+1}}^{\tau+n}$.

## :badminton:	Evaluation Metrics

$$ 
\begin{flalign}
&Score = min(l_1, l_2, ..., l_6)&
\end{flalign}
$$

$$ 
\begin{flalign}
l_i = AVG(CE + MAE)&
= \frac{ \sum_{r=1}^{|R|} \sum_{n=\tau+1}^{|r|} [S_n log \hat{S_n} + (|x_n-\hat{x_n}|+|y_n-\hat{y_n}|)]} {|R|\cdot(|r|-\tau)} &
\end{flalign}
$$


## :badminton:	Baseline: ShuttleNet
### Overview
ShuttleNet is the first turn-based sequence forecasting model containing two encoder-decoder modified Transformer as extractors, and a position-aware gated fusion network for fusing these contexts to tackle stroke forecasting in badminton.
Please refer to the [paper](https://ojs.aaai.org/index.php/AAAI/article/view/20341) for more details.
Here we adapt ShuttleNet to our newly collected dataset as the official baseline in the CoachAI Badminton Challenge.
All hyper-parameters are set as the same in the paper.

### Code Usage
#### Train a model
```=bash
./script.sh
```

#### Generate predictions
```=bash
python generator.py {model_path}
```

#### Run evaluation metrics
- Both ground truth and prediction files are default in the `data` folder
```=bash
python evaluation.py
```
