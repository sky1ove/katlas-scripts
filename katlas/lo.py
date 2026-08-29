"""calculation of log-odds PSSM and visualization"""


import numpy as np
from matplotlib import pyplot as plt

from .data import Data
from .plot import _plot_logo_heatmap_layout, convert_logo_df, plot_heatmap, plot_logo_raw
from .pssm import EPSILON, recover_pssm, get_prob, flatten_pssm

def get_pssm_LO(pssm_df,
                bg_type=None, # S, T, Y, ST, or STY
                bg_pssm=None, # Alternative: provide background PSSM directly
               ):
    "Get log odds PSSM: log2 (freq pssm/background pssm)."
    
    # Can't have both bg_pssm and bg_type
    if bg_pssm is not None and bg_type is not None:
        raise ValueError("Cannot specify both bg_pssm and bg_type")
    if bg_pssm is None and bg_type is None:
        raise ValueError("Must specify either bg_pssm or bg_type")
    
    # Get background PSSM
    if bg_type is not None:
        bg_pssms = Data.ks_background()
        flat_bg = bg_pssms.loc[f'ks_{bg_type}']
        bg_pssm = recover_pssm(flat_bg)
    
    # Compute log odds using log2 subtraction instead of division
    log_pssm = np.log2(pssm_df + EPSILON)
    log_bg = np.log2(bg_pssm + EPSILON)
    pssm_LO = log_pssm - log_bg
    # Cells with no data have undefined odds and otherwise collapse onto the EPSILON floor (the +-22
    # spikes that wash out the colour scale): freq == 0 -> -inf side, bg == 0 -> +inf side. Leave them
    # NaN so they render blank in the heatmap (the logo path fills NaN with 0, i.e. simply no letter).
    pssm_LO = pssm_LO.mask((pssm_df == 0) | (bg_pssm == 0))

    # make sure all columns and index matched
    if pssm_LO.shape != pssm_df.shape: raise ValueError("Shape mismatch between PSSM and background PSSM.")
    return pssm_LO

def get_pssm_LO_flat(flat_pssm,
                    bg_type=None, # S, T, Y, ST, or STY
                    bg_pssm=None,
                    ):
    pssm_df = recover_pssm(flat_pssm)
    return get_pssm_LO(pssm_df, bg_type=bg_type, bg_pssm=bg_pssm)

def plot_logo_LO(pssm_LO,title="Motif", acceptor=None, scale_zero=True,scale_pos_neg=True,ax=None,figsize=(10,1)):
    "Plot logo of log-odds given a frequency PSSM."
    if acceptor is not None:
        acceptor_upper = acceptor.upper()
        if acceptor_upper not in ["S", "T", "Y"]:
            raise ValueError(f"Acceptor must be one of 'S', 'T', or 'Y'; got {acceptor_upper!r}")
        pssm_LO = pssm_LO.copy()
        target_row = acceptor_upper.lower() if acceptor_upper.lower() in pssm_LO.index else acceptor_upper
        if target_row not in pssm_LO.index:
            pssm_LO.loc[target_row] = 0.0
        pssm_LO.loc[target_row, 0] = 0.1

    pssm_LO = convert_logo_df(pssm_LO,scale_zero=scale_zero,scale_pos_neg=scale_pos_neg)
    ytitle = "Scaled Log-Odds" if scale_pos_neg else "Log-Odds (bits)"
    plot_logo_raw(pssm_LO,ax=ax,title=title,ytitle=ytitle,figsize=figsize)

def plot_logo_heatmap_LO(pssm_LO, # pssm of log-odds
                             title='Motif',
                         acceptor=None,
                             figsize=(17,10),
                             include_zero=False,
                         scale_pos_neg=True,
                         square=False, # auto-size the width so heatmap cells render square
                         hspace=0.11, # vertical gap between the logo and the heatmap
                      ):
    """Plot logo and heatmap of enrichment bits vertically"""
    _plot_logo_heatmap_layout(
        heatmap_df=pssm_LO,
        logo_plotter=lambda ax: plot_logo_LO(
            pssm_LO,
            acceptor=acceptor,
            ax=ax,
            title=title,
            scale_pos_neg=scale_pos_neg,
        ),
        figsize=figsize,
        include_zero=include_zero,
        square=square,
        hspace=hspace,
        heatmap_kwargs={"scale_pos_neg": scale_pos_neg, "colorbar_title": "bits"},
    )
