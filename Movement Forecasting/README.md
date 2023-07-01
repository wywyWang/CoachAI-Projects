# Where Will Players Move Next? Dynamic Graphs and Hierarchical Fusion for Movement Forecasting in Badminton (AAAI 2023)

Official code of the paper Where Will Players Move Next? Dynamic Graphs and Hierarchical Fusion for Movement Forecasting in Badminton. Paper: https://arxiv.org/abs/2211.12217.

## Overview
Sports analytics has captured increasing attention since analysis of the various data enables insights for training strategies, player evaluation, etc.
In this paper, we focus on predicting what types of returning strokes will be made, and where players will move to based on previous strokes.
As this problem has not been addressed to date, movement forecasting can be tackled through sequence-based and graph-based models by formulating as a sequence prediction task.
However, existing sequence-based models neglect the effects of interactions between players, and graph-based models still suffer from multifaceted perspectives on the next movement.
Moreover, there is no existing work on representing strategic relations among players' shot types and movements.
To address these challenges, we first introduce the procedure of the Player Movements (PM) graph to exploit the structural movements of players with strategic relations.
Based on the PM graph, we propose a novel Dynamic Graphs and Hierarchical Fusion for Movement Forecasting model (DyMF) with interaction style extractors to capture the mutual interactions of players themselves and between both players within a rally, and dynamic players' tactics across time.
In addition, hierarchical fusion modules are designed to incorporate the style influence of both players and rally interactions.

## Model Framework
<img width="624" alt="Model framework" src="./model-architecture.png">

## Setup
- Build environment
  ```
  conda env create -f environment.yml
  ```
- Put data in folder `data/`

## Script
- To run multiple models:  
  The script would run all the models with different encode lengths and evaluate the models
   ```
  ./experiment.sh
  ```  
  Change the list in script to what models you want to run
  ```
  declare -a model_list=("Model_1" "Model_2" "Model_3" "Model_4")
  ```
## Code
- Train:  
  There are other parameters can change, please refer to the code
  ```
  python train.py --model_type ${model_type} --encode_length ${encode_length}
  ```
- Evaluate:  
  `train.py` would output the model folder path
  ```
   python evaluate.py ${model_folder_path}
   ```
   
## Citation
If you use our dataset or find our work is relevant to your research, please cite:
```
@inproceedings{DyMF_Badminton, 
    author    = {Kai{-}Shiang Chang and
               Wei{-}Yao Wang and
               Wen{-}Chih Peng},
    title={Where Will Players Move Next? Dynamic Graphs and Hierarchical Fusion for Movement Forecasting in Badminton}, 
    publisher = {{AAAI} Press},
    booktitle = {{AAAI}},
    year={2023},
    pages={6998-7005} 
}
```