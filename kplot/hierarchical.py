"""Plots for hierarchical clustering"""


import pandas as pd,numpy as np,seaborn as sns
from concurrent.futures.process import BrokenProcessPool
from tqdm import tqdm
from functools import partial
from tqdm.contrib.concurrent import process_map
from fastcore.meta import delegates
from scipy.cluster.hierarchy import fcluster,linkage,dendrogram
from scipy.spatial.distance import pdist, euclidean
from matplotlib import pyplot as plt
from cachetools import LRUCache

def get_1d_distance(df,func_flat):
    "Compute 1D distance (like pdist from scipy) but for df with column names"
    n = len(df)
    dist = []
    for i in tqdm(range(n)):
        for j in range(i+1, n):
            d = func_flat(df.iloc[i], df.iloc[j])
            dist.append(d)
    return np.array(dist)

def get_distance(pair, df, func):
    i,j=pair
    return func(df.iloc[i], df.iloc[j])

def get_1d_distance_parallel(df, func_flat, max_workers=4, chunksize=100):
    "Parallel compute 1D distance for each row in a dataframe given a distance function "
    n = len(df)
    index_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    bound_worker = partial(get_distance, df=df, func=func_flat)

    dist = process_map(bound_worker, index_pairs, max_workers=max_workers, chunksize=chunksize)
    return np.array(dist)

_Z_CACHE = LRUCache(maxsize=50)

def get_Z(pssms,func_flat=euclidean,method='ward',parallel=True):
    "Get linkage matrix Z from pssms dataframe"
    key = id(pssms)
    if key not in _Z_CACHE:
        if func_flat is euclidean:
            distance = pdist(pssms.fillna(0).to_numpy(), metric="euclidean")
        else:
            try:
                distance = get_1d_distance_parallel(pssms,func_flat=func_flat) if parallel else get_1d_distance(pssms,func_flat=func_flat)
            except BrokenProcessPool:
                distance = get_1d_distance(pssms,func_flat=func_flat)
        Z = linkage(distance, method=method)
        _Z_CACHE[key] =Z
    return _Z_CACHE[key]

def plot_dendrogram(Z,
                    thr=0.07,
                    dense=4, # the higher the more dense for each row
                    line_width=1,
                    title=None,
                    scale=1,
                    **kwargs):
    length = (len(Z) + 1) // dense
    
    plt.figure(figsize=(5*scale,length*scale))
    with plt.rc_context({'lines.linewidth': line_width}):
        dendrogram(
            Z,
            orientation='left',
            leaf_font_size=7,
            color_threshold=thr,
            **kwargs
        )
    if title is not None: plt.title(title)
    plt.xlabel('Distance')
    # plt.savefig(output, bbox_inches='tight')
    # plt.close()
    ax = plt.gca()
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

