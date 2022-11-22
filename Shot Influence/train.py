"""Utility functions."""
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

import util


def split_data(dataset: pd.DataFrame,
               val_ratio: int = 0.2,
               test_mask: List[bool] = None
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into training, validation and testing set."""
    test = dataset[test_mask]
    non_test = dataset[~test_mask]
    groups = non_test['rally_id'].unique()
    ngroup = len(groups)
    val_groups = np.random.choice(groups, int(ngroup * val_ratio))
    val_mask = non_test['rally_id'].isin(val_groups)
    val = non_test[val_mask]
    train = non_test[~val_mask]
    return train, val, test


def prepare_data(dataset: pd.DataFrame,
                 shot_attributes: Union[str, List[str], List[Union[str, List[str]]]],
                 rally_attributes: Union[str, List[str], List[Union[str, List[str]]]],
                 pad_to: int,
                 min_len: int = 1
                 ) -> Tuple[Union[np.ndarray, List[np.ndarray]],
                            Union[np.ndarray, List[np.ndarray]]]:
    """Convert dataset to appropriate format for training."""
    shots = []
    rallies = []
    shot_attributes_f = util.flatten(shot_attributes)
    rally_attributes_f = util.flatten(rally_attributes)
    
    # Generate sequences of rallies
    for rally_id, rally in dataset.groupby('rally_id'):
        if min_len > 0 and len(rally) < min_len:
            continue
        shots.append(rally[shot_attributes_f].values.astype('float32'))
        rallies.append(rally[rally_attributes_f].values[-1].astype('float32'))
        # Force non-target's sequence starts at second step
        pad = ((0, pad_to - len(rally)) if rally['is_target_turn'].iloc[0]
               else (1, pad_to - len(rally) - 1))
        shots[-1] = np.pad(shots[-1], [pad, (0, 0)])
    shots = np.asarray(shots)
    rallies = np.asarray(rallies)

    # Split back to input specification
    shot_attributes_len = util.list_len(shot_attributes)
    if len(shot_attributes_len) > 1:
        shots = np.split(shots, np.cumsum(shot_attributes_len)[:-1], axis=-1)
        # Reduce dimension when there is one attribute only
        for i in range(len(shots)):
            if shot_attributes_len[i] == 1:
                shots[i] = shots[i][:, :, 0]
    rally_attributes_len = util.list_len(rally_attributes)
    if len(rally_attributes_len) > 1:
        rallies = np.split(rallies, np.cumsum(rally_attributes_len)[:-1], axis=-1)
        for i in range(len(rallies)):
            if rally_attributes_len[i] == 1:
                rallies[i] = rallies[i][:, 0]
    return (shots, rallies)
