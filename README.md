# CoachAI-Projects

## Overview
This repo contains official implementations of **Coach AI Badminton Project** from [Advanced Database System Laboratory, National Yang Ming Chiao Tung University](https://sites.google.com/view/nycu-adsl) supervised by [Prof. Wen-Chih Peng](https://sites.google.com/site/wcpeng/).

The high-level concepts of each project are as follows:
1. [Visualization Platform](https://github.com/wywyWang/CoachAI-Projects/tree/main/Visualization%20Platform) published at *Physical Education Journal 2020* aims to construct a platform that can be used to illustrate the data from matches.
2. [Shot Influence and Extension Work](https://github.com/wywyWang/CoachAI-Projects/tree/main/Shot%20Influence) published at *ICDM-21* and *ACM TIST 2022*, respectively introduce a framework with a shot encoder, a pattern extractor, and a rally encoder to capture long short-term dependencies for evaluating players' performance of each shot. 
3. [Stroke Forecasting](https://github.com/wywyWang/CoachAI-Projects/tree/main/Stroke%20Forecasting) published at *AAAI-22* proposes the first stroke forecasting task to predict the future strokes of both players based on the given strokes by ShuttleNet, a position-aware fusion of rally progress and player styles framework.
4. [Strategic Environment](https://github.com/wywyWang/CoachAI-Projects/tree/main/Strategic%20Environment) published at *AAAI-23 Student Abstract* designs a safe and reproducible badminton environment for turn-based sports, which simulates rallies with different angles of view and designs the states, actions, and training procedures.
5. [Movement Forecasting](https://github.com/wywyWang/CoachAI-Projects/tree/main/Movement%20Forecasting) published at *AAAI-23* proposes the first movement forecasting task, which contains not only the goal of stroke forecasting but also the movement of players, by DyMF, a novel dynamic graphs and hierarchical fusion model based on the proposed player movements (PM) graphs.
6. [CoachAI-Challenge-IJCAI2023](https://github.com/wywyWang/CoachAI-Projects/tree/main/CoachAI-Challenge-IJCAI2023) is a badminton challenge (CC4) hosted at *IJCAI-23*. Please find the [website](https://sites.google.com/view/coachai-challenge-2023/) for more details.
7. [ShuttleSet](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet) published at *KDD-23* is the largest badminton singles dataset with stroke-level records.
    - An extension dataset [ShuttleSet22](https://github.com/wywyWang/CoachAI-Projects/tree/main/CoachAI-Challenge-IJCAI2023/ShuttleSet22) published at *IJCAI-24 Demo & IJCAI-23 IT4PSS Workshop* is also released.
8. [CoachAI Badminton Environment](https://github.com/wywyWang/CoachAI-Projects/tree/main/CoachAI%20Badminton%20Environment) published at *AAAI-24 Student Abstract and Demo, DSAI4Sports @ KDD 2023* is a reinforcement learning (RL) environment tailored for AI-driven sports analytics, offering:  i) Realistic opponent simulation for RL training; ii) Visualizations for evaluation; and iii) Performance benchmarks for assessing agent capabilities. 

## Publications and Contributions
1. Wei-Yao Wang, Wen-Chih Peng, Wei Wang, Philip Yu, "ShuttleSHAP: A Turn-Based Feature Attribution Approach for Analyzing Forecasting Models in Badminton", [paper](https://arxiv.org/abs/2312.10942)
2. Wei-Yao Wang, Wei-Wei Du, Wen-Chih Peng, "Benchmarking Stroke Forecasting with Stroke-Level Badminton Dataset", IJCAI 2024 Demo & IT4PSS @ IJCAI 2023, [paper](https://arxiv.org/abs/2306.15664)
3. Kuang-Da Wang, Yu-Tse Chen, Yu-Heng Lin, Wei-Yao Wang, Wen-Chih Peng, "The CoachAI Badminton Environment: Bridging the Gap Between a Reinforcement Learning Environment and Real-World Badminton Games", AAAI 2024 Demo, [paper](https://ojs.aaai.org/index.php/AAAI/article/view/30584)
4. Kuang-Da Wang, Wei-Yao Wang, Yu-Tse Chen, Yu-Heng Lin, Wen-Chih Peng, "The CoachAI Badminton Environment: A Novel Reinforcement Learning Environment with Realistic Opponents (Student Abstract)", AAAI 2024, [paper](https://ojs.aaai.org/index.php/AAAI/article/view/30523)
5. Kuang-Da Wang, Wei-Yao Wang, Ping-Chun Hsieh, Wen-Chih Peng, "Generating Turn-Based Player Behavior via Experience from Demonstrations", SPIGM @ ICML 2023, [paper](https://openreview.net/forum?id=9cuULoi7Ex)
6. Kuang-Da Wang, Yu-Tse Chen, Yu-Heng Lin, Wei-Yao Wang, Wen-Chih Peng, "The CoachAI Badminton Environment: Improving Badminton Player Tactics with A Novel Reinforcement Learning Environment", DSAI4Sports @ KDD 2023
7. Wei-Yao Wang, Yung-Chang Huang, Tsi-Ui Ik, Wen-Chih Peng, "ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis", KDD 2023, [paper](https://arxiv.org/abs/2306.04948)
8. Kai-Shiang Chang, Wei-Yao Wang, Wen-Chih Peng, "Where Will Players Move Next? Dynamic Graphs and Hierarchical Fusion for Movement Forecasting in Badminton", AAAI 2023, [paper](https://ojs.aaai.org/index.php/AAAI/article/view/25855)
9. Li-Chun Huang, Nai-Zen Hseuh, Yen-Che Chien, Wei-Yao Wang, Kuang-Da Wang, Wen-Chih Peng, "A Reinforcement Learning Badminton Environment for Simulating Player Tactics (Student Abstract), AAAI 2023, [paper](https://ojs.aaai.org/index.php/AAAI/article/view/26976)
10. Wei-Yao Wang, "Modeling Turn-Based Sequences for Player Tactic Applications in Badminton Matches", CIKM 2022, [paper](https://dl.acm.org/doi/abs/10.1145/3511808.3557820)
11. Wei-Yao Wang, Teng-Fong Chan, Wen-Chih Peng, Hui-Kuo Yang, Chih-Chuan Wang, Yao-Chung Fan, "How Is the Stroke? Inferring Shot Influence in Badminton Matches via Long Short-term Dependencies", ACM TIST 2022, [paper](https://dl.acm.org/doi/full/10.1145/3551391)
12. Wei-Yao Wang, Hong-Han Shuai, Kai-Shiang Chang, Wen-Chih Peng, "ShuttleNet: Position-aware Fusion of Rally Progress and Player Styles for Stroke Forecasting in Badminton", AAAI 2022, [paper](https://ojs.aaai.org/index.php/AAAI/article/view/20341)
13. Wei-Yao Wang, Teng-Fong Chan, Wen-Chih Peng, Hui-Kuo Yang, Chih-Chuan Wang, Yao-Chung Fan, "Exploring the Long Short-Term Dependencies to Infer Shot Influence in Badminton Matches", ICDM 2021, [paper](https://ieeexplore.ieee.org/document/9679184)
14. Wei-Yao Wang, Kai-Shiang Chang, Teng-Fong Chen, Chih-Chuan Wang, Wen-Chih Peng, Chih-Wei Yi, "Badminton Coach AI: A Badminton Match Data Analysis Platform Based on Deep Learning", Physical Education Journal 2020, [paper](https://www.airitilibrary.com/Publication/alDetailedMesh?docid=10247297-202006-202007060015-202007060015-201-213)

## References
If you use our dataset or find our project is relevant to your research, please bib format from [here](https://github.com/wywyWang/CoachAI-Projects/blob/main/CITATIONS.bib).

