"""Scoring functions to calculate kinase score based on substrate sequence"""


import numpy as np, pandas as pd
from typing import Callable
from functools import partial
from tqdm import tqdm

from .data import Data
from .utils import STY2sty, check_seq, check_seqs, pSTY2sty

EPSILON = 1e-8

def cut_seq(input_string: str, # site sequence
            min_position: int, # minimum position relative to its center
            max_position: int, # maximum position relative to its center
            ):
    
    "Extract sequence based on a range relative to its center position"
    
    # Find the center position of the string
    center_position = len(input_string) // 2

    # Calculate the start and end indices
    start_index = max(center_position + min_position, 0)  # Ensure start_index is not negative
    end_index = min(center_position + max_position + 1, len(input_string))  # Ensure end_index does not exceed string length

    # Extract and return the substring
    return input_string[start_index:end_index]

def get_dict(input_string:str, # phosphorylation site sequence
            ):
    "Get a dictionary of input string; no need for the star in the middle; make sure it is 15 or 10 length"

    center_index = len(input_string) // 2
    result = []

    for i, char in enumerate(input_string):
        position = i - center_index

        if char.isalpha():
            result.append(f"{position}{char}")

    return result

def multiply(values, # list of values, possibilities of amino acids at certain positions
                  kinase=None,
             num_aa=23, # number of amino acids, 23 for standard CDDM, 20 for all uppercase CDDM
            ):
    
    "Multiply the possibilities of the amino acids at each position in a phosphorylation site"
    

    # Using the logarithmic property: log(a*b) = log(a) + log(b)
    # Compute the sum of the logarithms of the values and the scale factor
    values = [v+EPSILON for v in values]
    log_sum = np.sum(np.log2(values)) + (len(values) - 1) * np.log2(num_aa)

    return log_sum

multiply_23 = partial(multiply,num_aa=23)

multiply_20 = partial(multiply,num_aa=20)

def multiply_pspa(values, kinase, num_aa_dict=None):
    "Multiply values, consider the dynamics of scale factor, which is PSPA random aa number."
    if num_aa_dict is None:
        num_aa_dict = Data.num_dict()
    # Check if any values are less than or equal to zero
    if np.any(np.array(values) == 0):
        return np.nan
    else:
        # Retrieve the divide factor from the dictionary
        divide_factor = num_aa_dict[kinase]

        # Using the logarithmic property: log(a*b) = log(a) + log(b)
        # Compute the sum of the logarithms of the values and the divide factor
        log_sum = np.sum(np.log2(values)) + (len(values) - 1) * np.log2(divide_factor)

        return log_sum

def sumup(values, # list of values, possibilities of amino acids at certain positions
          kinase=None,
         ):
    "Sum up the possibilities of the amino acids at each position in a phosphorylation site sequence"
    return sum(values)

def meanup(values, # list of per-position log-odds at the matched positions
           kinase=None,
          ):
    "Average (length-normalised sumup) the per-position log-odds across matched positions; NaN-safe."
    arr = np.asarray(values, dtype=float); arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else np.nan

def duplicate_ref_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    If '0S', '0T', '0Y' exist with non-zero values, create '0s', '0t', '0y' with same values.
    If '0s', '0t', '0y' exist with non-zero values, create '0S', '0T', '0Y' with same values.
    """
    df = df.copy()
    pairs = [('0S', '0s'), ('0T', '0t'), ('0Y', '0y')]

    for upper, lower in pairs:
        if upper in df.columns and (df[upper] != 0).any():
            df[lower] = df[upper]
        elif lower in df.columns and (df[lower] != 0).any():
            df[upper] = df[lower]

    return df

def preprocess_ref(ref):
    "Convert pS/T/Y in ref columns to s/t/y if any; mirror 0S/T/Y to 0s/t/y."
    ref = ref.copy()
    # if ref contains pS,pT,pY columns, convert them to s,t,y for scoring
    ref.columns=ref.columns.map(pSTY2sty)
    # duplicate 0S/T/Y to 0s/t/y (or the opposite) to ensure equal treatment of zero position
    return duplicate_ref_zero(ref)


def Params(name=None, load=True):
    def lazy(f): return lambda: f().astype('float32')
    
    params = {
        "CDDM": {'ref': lazy(Data.cddm), 'func': multiply_23},
        "CDDM_upper": {'ref': lazy(Data.cddm_upper), 'func': multiply_20, 'to_upper': True},
        "PSPA_st": {'ref': lazy(Data.pspa_st), 'func': multiply_pspa},
        "PSPA_y": {'ref': lazy(Data.pspa_tyr), 'func': multiply_pspa},
        "PSPA": {'ref': lazy(Data.pspa), 'func': multiply_pspa},
    }

    if name is None:
        return list(params.keys())

    cfg = params[name]
    if load and callable(cfg['ref']):
        cfg['ref'] = cfg['ref']()  # actually load now
    return cfg

def multiply_generic(merged_df, kinases, df_index, divide_factor_func):
    """Multiply-based log-sum aggregation across kinases."""
    out = {}
    log2 = np.log2  # local alias for speed
    
    for kinase in tqdm(kinases, desc="Computing multiply_generic"):
        divide_factor = divide_factor_func(kinase)
        df = merged_df[['input_index', kinase]].dropna()
        if df.empty:
            out[kinase] = pd.Series(index=df_index, dtype=float)
            continue
        
        log_values = log2(df[kinase] + EPSILON)
        grouped = df.assign(log_value=log_values).groupby('input_index')['log_value']
        
        # vectorized form
        log_sum = grouped.sum() + (grouped.count() - 1) * log2(divide_factor)
        out[kinase] = log_sum

    return pd.DataFrame(out).reindex(df_index)

def predict_kinase_df(df, seq_col, ref, func, to_lower=False, to_upper=False):
    """
    Predict kinase scores based on reference PSSM or weight matrix.
    Applies preprocessing, merges long format keys, then aggregates using given func.
    """
    print(f"Input dataframe has {len(df)} rows")
    print("Preprocessing...")

    ref = preprocess_ref(ref)
    df = df.copy()
    df[seq_col] = check_seqs(df[seq_col])

    if to_lower:
        df[seq_col] = df[seq_col].apply(STY2sty)
    if to_upper:
        df[seq_col] = df[seq_col].str.upper()

    pos = ref.columns.str[:-1].astype(int)
    df[seq_col] = df[seq_col].apply(partial(cut_seq, min_position=pos.min(), max_position=pos.max()))

    print("Preprocessing done. Expanding sequences...")

    input_keys_df = (
        df.assign(keys=df[seq_col].apply(get_dict))
          .explode("keys")
          .reset_index(names="input_index")[["input_index", "keys"]]
          .rename(columns={"keys": "key"})
          .set_index("key")
    )

    print("Merging reference...")
    ref_T = ref.T.astype("float32")
    merged_df = input_keys_df.merge(ref_T, left_index=True, right_index=True, how="inner")
    print("Merge complete.")

    if func == sumup:
        out = merged_df.groupby("input_index").sum().reindex(df.index)
    elif func == meanup:
        # groupby.mean() skips NaN per kinase column -> average over each kinase's matched/defined positions
        out = merged_df.groupby("input_index").mean().reindex(df.index)
    elif func in (multiply, multiply_pspa, multiply_23, multiply_20):
        num_dict = Data.num_dict() if func == multiply_pspa else None
        divisor = (
            (lambda k: num_dict[k])
            if func == multiply_pspa else
            (lambda k: 20 if func == multiply_20 else 23)
        )
        out = multiply_generic(merged_df, ref_T.columns, df.index, divide_factor_func=divisor)
    else:
        raise ValueError(f"Unknown function: {func}")

    return out.round(3)


def get_pct_df(score_df, # output from predict_kinase_df
               pct_ref, # a reference df for percentile calculation
              ):
    "Replicate the precentile results from The Kinase Library."

    percentiles = np.zeros(score_df.shape)

    for i, kinase in tqdm(enumerate(score_df.columns), total=len(score_df.columns)):
        ref_values = np.sort(pct_ref[kinase].values)
        scores = score_df[kinase].values

        less = np.searchsorted(ref_values, scores, side="left")
        equal_or_less = np.searchsorted(ref_values, scores, side="right")
        equal = equal_or_less - less

        percentiles[:, i] = (less + 0.5 * equal) / len(ref_values) * 100

    percentiles_df = pd.DataFrame(percentiles, index=score_df.index, columns=score_df.columns).astype(float).round(3)
    return percentiles_df
