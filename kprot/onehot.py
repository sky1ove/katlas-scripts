"""One-hot encoding and clustering helpers for aligned protein sequence windows."""


from collections.abc import Sequence

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

def onehot_encode(
    sequences: Sequence[str],  # aligned protein sequence windows
    transform_colname: bool = True,  # shift feature names around the center residue
    n: int = 20,  # center position used for transformed column labels
) -> pd.DataFrame:
    "One-hot encode aligned protein sequence windows."
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_array = encoder.fit_transform([list(seq) for seq in sequences])
    colnames = [name[1:] for name in encoder.get_feature_names_out()]
    if transform_colname:
        colnames = [f"{int(item.split('_', 1)[0]) - n}{item.split('_', 1)[1]}" for item in colnames]
    encoded_df = pd.DataFrame(encoded_array, columns=colnames)
    return encoded_df


def run_kmeans(
    onehot: pd.DataFrame,  # one-hot encoded sequence matrix
    n: int = 2,  # number of clusters to fit
    seed: int = 42,  # random seed for KMeans
) -> object:
    "Fit KMeans to one-hot encoded features and return the assigned labels."
    kmeans = KMeans(n_clusters=n, random_state=seed, n_init="auto")
    return kmeans.fit_predict(onehot)

def filter_range_columns(
    df: pd.DataFrame,  # one-hot encoded dataframe with position-prefixed column names
    low: int = -10,  # lower bound for retained positions
    high: int = 10,  # upper bound for retained positions
) -> pd.DataFrame:
    "Filter one-hot columns to a position window around the center residue."
    positions = df.columns.str[:-1].astype(int)
    mask = (positions >= low) & (positions <= high)
    return df.loc[:, mask]

