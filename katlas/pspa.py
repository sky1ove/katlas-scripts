"""PSPA data visualization"""


import numpy as np, pandas as pd
from matplotlib import pyplot as plt

from .plot import _plot_logo_heatmap_layout, plot_logo_raw, prepare_logo_df, scale_zero_position
from .pssm import recover_pssm

def preprocess_pspa(pssm):
    "Drop row s as it's a duplicate of t; rename t to pS/pT; calculate np.log2(pssm/pssm.median())"
    pssm = prepare_logo_df(pssm)
    pssm = pssm.drop(index='s')
    pssm.index = pssm.index.map(lambda x: x.replace('t','pS/pT').replace('y','pY'))
    # pssm = np.log2(pssm/pssm.median()) # have to do it without position 0
    non_zero_cols = pssm.columns != 0
    pssm.loc[:, non_zero_cols] = np.log2(
        pssm.loc[:, non_zero_cols] / pssm.loc[:, non_zero_cols].median()
    )
    pssm=scale_zero_position(pssm)
    return pssm


def plot_logo_heatmap_pspa(row, # row of Data.pspa()
                       title='Motif',
                       figsize=(4.5,10), # with square=True only the height is used; width is computed
                       include_zero=False,
                       colorbar_title='log₂', # short; pass '' to hide it (the logo already labels the scale)
                       square=True, # auto-size the figure width so heatmap cells come out square
                       hspace=0.11, # vertical gap between the logo and the heatmap
                       height_ratios=(1, 5), # logo:heatmap height split (raise 1st for a taller logo / flatter heatmap)
                      ):
    """Plot logo and heatmap vertically. The heatmap encodes log2(value/median), matching the logo.

    Note: we do NOT force seaborn `square=True` — that shrinks/re-centres the heatmap axis and breaks its
    alignment with the logo above. Instead, when `square=True`, we compute the figure WIDTH from the height and
    the matrix shape so each cell is square while the axis keeps spanning the logo's width (staying aligned).
    This adapts to the position count automatically: 10 for S/T kinases (-5..+4), 11 for Tyr kinases with +5.
    """
    pssm = recover_pssm(row.dropna())
    logo_pssm = preprocess_pspa(pssm)

    # Heatmap in the same log2(value/median) space as the logo. This mirrors
    # preprocess_pspa (drop duplicate s, rename t->pS/pT & y->pY, log2 vs the
    # per-position median) but WITHOUT the logo-only zero-position rescaling,
    # since position 0 is masked in the heatmap anyway.
    heat = prepare_logo_df(pssm).drop(index='s')
    heat.index = heat.index.map(lambda x: x.replace('t', 'pS/pT').replace('y', 'pY'))
    non_zero = heat.columns != 0
    heat.loc[:, non_zero] = np.log2(heat.loc[:, non_zero] / heat.loc[:, non_zero].median())

    _plot_logo_heatmap_layout(
        heatmap_df=heat,
        logo_plotter=lambda ax: plot_logo_raw(
            logo_pssm,
            ax=ax,
            ytitle='log₂(Value/Median)',
            title=title,
        ),
        figsize=figsize,
        include_zero=include_zero,
        square=square,
        hspace=hspace,
        height_ratios=height_ratios,
        heatmap_kwargs={"colorbar_title": colorbar_title},
    )


