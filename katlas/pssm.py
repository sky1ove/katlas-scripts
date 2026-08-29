"""Functions related with PSSMs"""


import numpy as np, pandas as pd
from fastcore.meta import delegates
from tqdm import tqdm
from typing import Sequence
from matplotlib import pyplot as plt

from .utils import check_seqs

EPSILON = 1e-8

def get_cnt(data: pd.DataFrame | pd.Series | Sequence[str], # input data, list or df
            seq_col: str='site_seq', # column name if input is df
            ):
    "Get the count matrix of amino acids at each position from phosphorylation site sequences."

    aa_order=[i for i in 'PGACSTVILMFYWHKRQNDEsty']

    if isinstance(data, pd.DataFrame):
        if seq_col not in data.columns:
            raise ValueError(f"Column '{seq_col}' not found in DataFrame.")
        site = data[seq_col]
    else:
        if isinstance(data, (str, bytes)):
            raise TypeError("Input looks like a single sequence string; pass [seq] or a Series instead.")
        try:
            site = pd.Series(data,copy=False)
        except Exception:
            raise TypeError("Input must be a DataFrame, Series, or list of sequences.")
    
    
    site = check_seqs(site)
    
    site_array = np.array(site.apply(list).tolist())
    seq_len = site_array.shape[1]

    position = list(range(-(seq_len // 2), (seq_len // 2)+1)) # add 1 because range do not include the final num
    
    site_df = pd.DataFrame(site_array, columns=position)
    melted = site_df.melt(var_name='Position', value_name='aa')
    
    grouped = melted.groupby(['Position', 'aa']).size().reset_index(name='Count')
    grouped = grouped[grouped.aa.isin(aa_order)].reset_index(drop=True)
    
    cnt_df = grouped.pivot(index='aa', columns='Position', values='Count').fillna(0).astype(int)

    ordered_aa = [aa for aa in aa_order if aa in cnt_df.index]
    cnt_df = cnt_df.reindex(index=ordered_aa, columns=position, fill_value=0)
    
    return cnt_df

def get_prob(data: pd.DataFrame | pd.Series | Sequence[str], # input data, list or df
             seq_col: str='site_seq', # column name if input is df
             ):
    "Get the probability matrix of PSSM from phosphorylation site sequences."

    cnt_df = get_cnt(data, seq_col)
    pssm_df = cnt_df / cnt_df.sum()
    
    return pssm_df


def flatten_pssm(pssm_df,
                 column_wise=True, # if True, column major flatten; else row wise flatten (for pytorch training)
                ):
    "Flatten PSSM dataframe to dictionary"
    
    pssm_df=pssm_df.copy()

    # Flatten a pssm_df
    # Column wise
    if column_wise: 
        pssm = pssm_df.unstack().reset_index(name='value')
        # Combine position column and residue identity column as new column for keys
        pssm['position_residue']=pssm.iloc[:,0].astype(str)+pssm.iloc[:,1]
        
    # Row wise
    else: 
        pssm = pssm_df.T.unstack().reset_index(name='value')
        pssm['position_residue']=pssm.iloc[:,1].astype(str)+pssm.iloc[:,0].astype(str)
    
    # Set index to be position+residue
    return pssm.set_index('position_residue')['value'].to_dict()

def recover_pssm(flat_pssm: pd.Series):
    "Recover 2D PSSM from flattened PSSM Series."
    df = flat_pssm.reset_index()
    df.columns=['index', 'value']
    df['Position'] = df['index'].str[:-1].astype(int)
    df['aa'] = df['index'].str[-1]

    df = df.pivot(index='aa', columns='Position', values='value').fillna(0)
    aa_order=tuple('PGACSTVILMFYWHKRQNDEsty')
    order = [aa for aa in aa_order if aa in df.index]
    return df.reindex(index=order).sort_index(axis=1)

def _clean_zero(pssm_df):
    "Zero out non-last three values in position 0 (keep only s,t,y values at center)"
    pssm_df = pssm_df.copy()
    standard_aa = list(set(pssm_df.index)-set(['s','t','y']))
    pssm_df.loc[standard_aa, 0] = 0
    return pssm_df

def clean_zero_normalize(pssm_df):
    "Zero out non-last three values in position 0 (keep only s,t,y values at center), and normalize per position"
    pssm_df=pssm_df.copy()
    pssm_df.columns= pssm_df.columns.astype(int)
    pssm_df = _clean_zero(pssm_df)
    return pssm_df/pssm_df.sum()

def pssm_to_seq(pssm_df, 
                thr=0.2, # threshold of probability to show in sequence
                clean_center=True, # if true, zero out non-last three values in position 0 (keep only s,t,y values at center)
                ):
    "Represent PSSM in string sequence of amino acids"
    
    pssm_df = pssm_df.copy()
    if clean_center:
        pssm_df.loc[pssm_df.index[:-3], 0] = 0  # keep only s,t,y in center 0 position

    pssm_df.index = pssm_df.index.map(lambda x: x.replace('pS', 's').replace('pT', 't').replace('pY', 'y'))

    consensus = []
    for i, col in enumerate(pssm_df.columns):
        # consider the case where sum for the position is 0
        column_vals = pssm_df[col]
        if column_vals.sum() == 0:
            symbol = '_'
        else:
            top = column_vals.nlargest(3)
            passing = [aa for aa, prob in zip(top.index, top.values) if prob > thr]

            if not passing:
                symbol = '.'
            elif len(passing) == 1:
                symbol = passing[0]
            else:
                symbol = f"[{'/'.join(passing)}]"
                
        if col == 0:  # center position
            if symbol.startswith('['):
                symbol = symbol[:-1] + ']*'
            else:
                symbol += '*'

        consensus.append(symbol)

    return ''.join(consensus)


def get_cluster_pssms(df, 
                    cluster_col, 
                    seq_col='site_seq', 
                    id_col = 'sub_site',
                    count_thr=10, # if less than the count threshold, not include in the return
                    valid_thr=None, # percentage of not-nan values in pssm
                      plot=False):
    "Extract motifs from clusters of sequences in a dataframe"
    pssms = []
    ids = []
    # drop duplicates based both on cluster column and substrate seq id column
    if id_col is not None: df = df.drop_duplicates(subset=[cluster_col,id_col]).copy()
    value_counts = df[cluster_col].value_counts()
    
    for cluster_id, counts in tqdm(value_counts.items(),total=len(value_counts)):
        if count_thr is not None and counts < count_thr:
            continue
            
        df_cluster = df[df[cluster_col] == cluster_id]
        n= len(df_cluster)
        pssm = get_prob(df_cluster, seq_col)
        valid_score = (pssm != 0).sum().sum() / (pssm.shape[0] * pssm.shape[1])

        if valid_thr is not None and valid_score <= valid_thr:
            continue

        pssms.append(flatten_pssm(pssm))
        ids.append(cluster_id)

        if plot:
            from katlas.plot import plot_logo
            plot_logo(pssm, title=f'Cluster {cluster_id} (n={n})', figsize=(14, 1))
            plt.show()
            plt.close()

    pssm_df = pd.DataFrame(pssms, index=ids)
    return pssm_df

def get_entropy(pssm_df,# a dataframe of pssm with index as aa and column as position
            return_min=False, # return min entropy as a single value or return all entropy as a pd.series
            exclude_zero=False, # exclude the column of 0 (center position) in the entropy calculation
            clean_zero=True, # if true, zero out non-last three values in position 0 (keep only s,t,y values at center)
            ): 
    "Calculate entropy per position of a PSSM surrounding 0. The less entropy the more information it contains."
    pssm_df = pssm_df.copy()
    pssm_df.columns= pssm_df.columns.astype(int)
    if 0 in pssm_df.columns:
        if clean_zero:                       
            pssm_df = _clean_zero(pssm_df)
        if exclude_zero:
            # remove columns starts with zero and columns with interger name 0
            cols_to_drop = [col for col in pssm_df.columns 
                            if col == 0 or (isinstance(col, str) and col.startswith('0'))]
            if cols_to_drop: pssm_df = pssm_df.drop(columns=cols_to_drop)

    pssm_df = pssm_df/pssm_df.sum()
    per_position = -np.sum(pssm_df * np.log2(pssm_df + EPSILON), axis=0)
    per_position[pssm_df.sum() == 0] = 0
    return float(per_position.min()) if return_min else per_position


@delegates(get_entropy)
def get_IC(pssm_df,**kwargs):
    """
    Calculate the information content (bits) from a frequency matrix,
    using log2(3) for the middle position and log2(len(pssm_df)) for others.
    The higher the more information it contains.
    """
    
    keep_center = kwargs.pop('keep_center', False)   # our own flag; not for get_entropy
    entropy_position = get_entropy(pssm_df,**kwargs)

    max_entropy_array = pd.Series(np.log2(len(pssm_df)), index=entropy_position.index)

    # set exclude_zero to False
    exclude_zero = kwargs.get('exclude_zero', False)
    # guard on 0 being a real position: assigning by label to a missing one would APPEND a new
    # entry, leaving max_entropy_array longer than entropy_position and breaking the mask below.
    # A flank-only matrix (position 0 dropped) is a legitimate input.
    if exclude_zero is False and 0 in max_entropy_array.index:
        max_entropy_array[0] = np.log2(3)

    # information_content = max_entropy - entropy --> log2(N) - entropy
    IC_position = max_entropy_array - entropy_position

    # a fully-conserved position (entropy 0) carries no *variation*, so it is normally blanked; this
    # is what hides a mixed S/T centre only when it varies. keep_center=True keeps position 0 so a
    # fixed centre residue (e.g. the surface-display pTyr) still shows.
    drop = entropy_position == 0
    if keep_center:
        drop = drop & (entropy_position.index != 0)
    IC_position[drop] = 0
    return IC_position


