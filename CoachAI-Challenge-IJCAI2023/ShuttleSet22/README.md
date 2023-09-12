# ShuttleSet22: Benchmarking Stroke Forecasting with Stroke-Level Badminton Dataset (IT4PSS @ IJCAI-23)
Official dataset of the paper **ShuttleSet22: Benchmarking Stroke Forecasting with Stroke-Level Badminton Dataset**.
Paper: https://arxiv.org/abs/2306.15664

## Dataset Descriptions
ShuttleSet22 is an extension from [ShuttleSet](https://arxiv.org/abs/2306.04948). ShuttleSet22 contains 30,172 strokes(2,888 rallies) in the training set, 1,400 strokes (450 rallies) in the validation set, and 2,040 strokes (654 rallies) in the testing set in 2022 with 35 top-ranking men's singles and women's singles players.

### match.csv
* video: lists the folder name in set/.
* id: serial number of matches.
* set: the number of played sets in a match (either 2 or 3).
* duration: the duration of a match in minutes.
* winner: the winner player.
* loser: the lose player.

### homography.csv
* homography_matrix: can be used to transform coordinates from the real-world system back to the camera system by $p=H^{-1}p’$.
* upleft_x,upright_x,downleft_x,downright_x,upleft_y,upright_y,downleft_y,downright_y: the four coordinates of the court in the camera coordinate system.

### set{1, 2, 3}.csv in each folder
* rally: serial number of rallies in a match
* ball_round: the order of the shot in a rally
* time (hr:min:sec): the shot’s hitting time
* frame_num: sec * fps = frame_num
* roundscore_A: Player A’s current score in the set
* roundscore_B: Player B’s current score in the set
* player: the player who performed the shot (A is the player winning the match, B otherwise)
* type: the type of shot (total 18). Please find the translation below.
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

### Shot type translation
| English | Chinese |
| ------- | ------- |
| net shot | 放小球 |
| return net | 擋小球 |
| smash | 殺球 |
| wrist smash | 點扣 |
| lob | 挑球 |
| defensive return lob | 防守回挑 |
| clear | 長球 |
| drive | 平球 |
| driven flight | 小平球 |
| back-court drive | 後場抽平球 |
| drop | 切球 |
| passive drop | 過渡切球 |
| push | 推球 |
| rush | 撲球 |
| defensive return drive | 防守回抽 |
| cross-court net shot | 勾球 |
| short service | 發短球 |
| long service | 發長球 |

## Citation
If you use our dataset or find our work is relevant to your research, please cite:
```
@article{ShuttleSet22,
  author    = {Wei{-}Yao Wang and
               Wei{-}Wei Du and
               Wen{-}Chih Peng},
  title     = {ShuttleSet22: Benchmarking Stroke Forecasting with Stroke-Level Badminton Dataset},
  journal   = {CoRR},
  volume    = {abs/2306.15664},
  year      = {2023}
}
```