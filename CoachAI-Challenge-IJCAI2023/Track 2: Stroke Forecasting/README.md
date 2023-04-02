# Track 2: Forecasting Future Turn-Based Strokes in Badminton Rallies

## Task Introduction
TBD

## Data Overview

## Problem Definition
TBD

## Evaluation Metrics
TBD

## Baseline: ShuttleNet
### Overview
ShuttleNet is the first turn-based sequence forecasting model containing two encoder-decoder modified Transformer as extractors, and a position-aware gated fusion network for fusing these contexts to tackle stroke forecasting in badminton.
Please refer to the [paper](https://ojs.aaai.org/index.php/AAAI/article/view/20341) for more details.
Here we adapt ShuttleNet to our newly collected dataset as the official baseline in the CoachAI Badminton Challenge.
All hyper-parameters are set as the same in the paper.

### Code Usage
For more details: TBD
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