# RallyNet: Offline Imitation of Turn-Based Player Behavior via Experiential Contexts and Brownian Motion (ECML PKDD 2024)

Official code of the paper RallyNet: Offline Imitation of Turn-Based Player Behavior via Experiential Contexts and Brownian Motion. Paper: https://arxiv.org/abs/2403.12406

## Overview
This is the official PyTorch implementation of RallyNet. The code includes the core RallyNet algorithm as well as the related experiments on the badmintton sample dataset [1][2].

## Set up
1. Build environment
```
conda env create -f environment.yml
```
2. Put data in Data/{K_fold}/
* Each rally will be split into individual trajectories (state-action pairs) based on the player. The serving player is defined as "player," while the receiving player is defined as "opponent." We have previded training and testing data (when K_fold = 0), so there are four files available (player_train.pkl, player_test.pkl, opponent_train.pkl, and opponent_test.pkl).
* The state space has a dimension of 17, which includes information such as score, player positions, ball position, opponent movement, and player numbers. The action space has a dimension of 5, which includes shot type, landing position, and moving position.
3. Build experience extracting dictionary. Dictionaries will be saved in the folder Data/{K_fold}/experience/
```
python building_experience_dicts.py --experience_from {player_training_rallies}
```
## Training
In order to train an agent with RallyNet run the following command:
```
python train.py --output_folder_name {output_folder_name} --player_train {player_training_rallies} --player_test {player_testing_rallies}  --opponent_train {opponent_training_rallies}  --opponent_test {opponent_testing_rallies}
```
For example:
```
python train.py --output_folder_name 46_g5_a05_testing --player_train player_train_0.pkl --player_test player_test_0.pkl --opponent_train opponent_train_0.pkl --opponent_test opponent_test_0.pkl
```
## Evaluation
To test your model's ability to recover turn-based player behavior on the dataset, run the following code:
```
python evaluate.py {model_path} {model_epoch} {given_or_not}
```
For example:
```
python evaluate.py 46_g5_a05_testing 0 False
```
This command evaluates the effectiveness of recovering the entire rally using the weights from the first epoch of RallyNet in "46_g5_a05_testing" folder.

## Output files
* Trained models will be saved in the folder Results/saved_model/{output_folder_name}/
* The performance of every training epochs will also be contained in folder Results/training_results/ and {output_folder_name}.txt

## Citation
If you use our dataset or find our work is relevant to your research, please cite:
```
@article{DBLP:journals/corr/abs-2403-12406,
  author       = {Kuang{-}Da Wang and
                  Wei{-}Yao Wang and
                  Ping{-}Chun Hsieh and
                  Wen{-}Chih Peng},
  title        = {Offline Imitation of Badminton Player Behavior via Experiential Contexts
                  and Brownian Motion},
  journal      = {CoRR},
  volume       = {abs/2403.12406},
  year         = {2024}
}
```

## References
[1] Wang et al. "ShuttleNet: Position-aware Rally Progress and Player Styles Fusion for Stroke Forecasting in Badminton." AAAI'22.
[2] Chang et al. "Where Will Players Move Next? Dynamic Graphs and Hierarchical Fusion for Movement Forecasting in Badminton." AAAI'23.
