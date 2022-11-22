# Shot Influence

## Introduction
This repository contains the source code for the paper accepted in ICDM'21: *Exploring the Long Short-Term Dependencies to Infer Shot Influence in Badminton Matches* and the extended approach accepted in ACM TIST 2022: *How Is the Stroke? Inferring Shot Influence in Badminton Matches via Long Short-term Dependencies*.

## Getting started
### Prerequisites
- python3==3.6.9
- numpy==1.17.2
- pandas==0.24.2
- tensorflow-gpu==2.0.0
- keras_self_attention
- [ProSeNet](https://github.com/rgmyr/tf-ProSeNet)
- [DeepMoji](https://github.com/bfelbo/DeepMoji)
- [ON-LSTM](https://github.com/CyberZHG/keras-ordered-neurons)
- [Transformer](https://github.com/CyberZHG/keras-transformer)

### Install
Download the project
- with CMD
```
git clone https://github.com/yao0510/Shot-Influence.git
```

## Usage
### Train a model
```=python
python training.py
```

### Evaluate a model
```=python
python evaluate.py
```

## Citation
If you use our dataset or find our work is relevant to your research, please cite:
```
@inproceedings{DBLP:conf/icdm/WangCYWFP21,
  author    = {Wei{-}Yao Wang and
               Teng{-}Fong Chan and
               Hui{-}Kuo Yang and
               Chih{-}Chuan Wang and
               Yao{-}Chung Fan and
               Wen{-}Chih Peng},
  title     = {Exploring the Long Short-Term Dependencies to Infer Shot Influence
               in Badminton Matches},
  booktitle = {{ICDM}},
  pages     = {1397--1402},
  publisher = {{IEEE}},
  year      = {2021}
}
```