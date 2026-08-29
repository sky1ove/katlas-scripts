"""Generic helpers for tabular multi-output training and scoring, with runnable examples built from a seaborn dataset."""


from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn import set_config
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

set_config(transform_output="pandas")

def get_splits(
    df: pd.DataFrame,
    stratified: str | None = None,  # col used for stratified sampling
    group: str | None = None,       # col used to keep grouped rows together
    nfold: int = 5,
    seed: int = 123,
) -> list[tuple[np.ndarray, np.ndarray]]:
    "Split samples in a dataframe with stratified, grouped, or stratified-grouped K-fold logic."

    def _log(colname: str) -> None:
        print(kf)
        split = splits[0]
        print(f"# {colname} in train set: {df.iloc[split[0]][colname].nunique()}")
        print(f"# {colname} in test set: {df.iloc[split[1]][colname].nunique()}")

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    if stratified is not None and group is None:
        kf = StratifiedKFold(nfold, shuffle=True, random_state=seed)
        for split in kf.split(df.index, df[stratified]):
            splits.append(split)
        _log(stratified)
    elif group is not None and stratified is None:
        kf = GroupKFold(nfold)
        for split in kf.split(df.index, groups=df[group]):
            splits.append(split)
        _log(group)
    elif stratified is not None and group is not None:
        kf = StratifiedGroupKFold(nfold, shuffle=True, random_state=seed)
        for split in kf.split(df.index, groups=df[group], y=df[stratified]):
            splits.append(split)
        _log(stratified)
    else:
        raise ValueError("Either 'stratified' or 'group' argument must be provided.")
    return splits

