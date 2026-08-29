"""motif_09b · Method-agnostic per-kinase specificity (sharpness) metrics.

motif_09 scores specificity as max per-position information content — but IC needs a per-position
DISTRIBUTION (sum to 1), so it works only for PSPA-scaled and CDDM-frequency, not the signed MLP-attr.
This measures the same intuition — is the motif *peaked* (a few strong determinants) or *flat*
(promiscuous)? — with SCALE-FREE concentration metrics computed on each method's RAW values, so PSPA
(raw), CDDM (frequency) and MLP-attr (signed) are all handled on the same footing, no sum-to-1 needed.

Every metric is over the flank (positions −5..+5, acceptor 0 excluded), on the per-position CENTERED
values `D = value − per-position mean` (so a distribution's deviation-from-flat and a signed
attribution's deviation-from-0 are treated alike). Concentration/peakiness is scale-free, so the raw
magnitudes never need normalising:

  gini          Gini of |D| over all flank cells — 0 = flat, →1 = one cell dominates. ↑ = sharp.
  kurtosis      excess kurtosis of D — heavy tails = a few strong outliers. ↑ = sharp.
  top3_share    share of total |D| carried by the 3 strongest cells. ↑ = sharp.
  sharp_mean    mean over flank positions of the per-position Gini(|D_p|) — typical residue
                selectivity within a position. ↑ = sharp.
  sharp_max     the single most selective position's Gini(|D_p|). ↑ = sharp.
  eff_positions participation ratio of per-position importance impₚ = Σ|D_p|: (Σimp)²/Σimp². The
                effective number of positions carrying the signal. ↓ = concentrated in a few positions.
  spec_index    composite headline: mean of the per-method z-scores of {gini, kurtosis, top3_share,
                sharp_mean, sharp_max, −eff_positions}. One sharpness score per (kinase, method).

Residues: PSPA drops the `_TYR` dual-specificity duplicates and the flanking `s` row (the array
measures pT, so off-centre s == t); CDDM / MLP-attr keep pS/pT distinct.

Inputs   kdata: pspa (raw), cddm, kinase_info; out/mlp_attr_pssm_full.parquet (motif_17)
Outputs  out/specificity_metrics.parquet (+ .csv) — long: (kinase, group, method, <metrics>)
         fig/spec_metrics_bygroup.svg, spec_metrics_bar_<method>.svg, spec_metrics_rank.svg,
         fig/spec_metrics_cross_method.svg, spec_metrics_shape.svg (PSPA),
         fig/spec_agreement_vs_index.svg (CDDM–PSPA agreement vs sharpness — supersedes motif_15)

Run:  python nbs/motif_09b_specificity_metrics.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paths import FIG, OUT
from scipy.stats import kurtosis as excess_kurtosis
from scipy.stats import spearmanr

import kdata
from katlas.pssm import recover_pssm
from katlas.utils import group_color
from kplot.bar import plot_bar, plot_group_bar
from kplot.ranking import plot_rank
from kplot.scatter import plot_rel
from kplot.utils import save_svg, set_sns

FLANK = [p for p in range(-5, 6) if p != 0]        # flank positions; acceptor 0 excluded
AA20 = list('ACDEFGHIKLMNPQRSTVWY')                # the residues CDDM and PSPA share for agreement
#: the composite spec_index averages the per-method z-scores of these (eff_positions enters negated)
COMPOSITE_UP = ['gini', 'kurtosis', 'top3_share', 'sharp_mean', 'sharp_max']
METHOD_ORDER = ['PSPA', 'CDDM', 'MLP-attr']
MLP_ATTR = OUT / 'mlp_attr_pssm_full.parquet'


def gini(x):
    "Gini coefficient of non-negative values; scale-free (0 = equal, →1 = one value dominates)."
    x = np.sort(np.abs(np.asarray(x, float)))
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.sum() == 0:
        return np.nan
    n = len(x)
    return float((2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n)


def flank_matrix(flat, method):
    "Recover a flat PSSM to residue × flank-position, with per-method residue handling."
    m = recover_pssm(flat.dropna())
    cols = [c for c in m.columns if c in FLANK]        # keep flank positions only
    m = m[cols]
    if method == 'PSPA' and 's' in m.index:
        m = m.drop(index='s')                          # array measures pT; off-centre s == t
    return m


def kinase_metrics(M):
    "Scale-free sharpness metrics for one kinase's residue × flank-position matrix."
    D = M.sub(M.mean(axis=0), axis=1)                  # deviation from the per-position mean
    d = D.to_numpy(float).ravel()
    d = d[np.isfinite(d)]
    ad = np.abs(d)
    if ad.sum() == 0 or len(ad) < 4:
        return None
    per_pos_gini = D.apply(lambda col: gini(np.abs(col.to_numpy(float))), axis=0).dropna()
    imp = np.abs(D).sum(axis=0).to_numpy(float)        # per-position importance Σ|D_p|
    imp = imp[np.isfinite(imp) & (imp > 0)]
    top3 = np.sort(ad)[::-1][:3]
    return {
        'gini': gini(ad),
        'kurtosis': float(excess_kurtosis(d, fisher=True, bias=False)),
        'top3_share': float(top3.sum() / ad.sum()),
        'sharp_mean': float(per_pos_gini.mean()) if len(per_pos_gini) else np.nan,
        'sharp_max': float(per_pos_gini.max()) if len(per_pos_gini) else np.nan,
        'eff_positions': float(imp.sum() ** 2 / np.square(imp).sum()) if len(imp) else np.nan,
    }


def build_method(name, flat_df, group_map):
    "Per-kinase metric table for one method + the per-method-standardised composite spec_index."
    rows = []
    for k in flat_df.index:
        if group_map.get(k) is None:
            continue
        M = flank_matrix(flat_df.loc[k], name)
        if M.shape[1] == 0:
            continue
        rec = kinase_metrics(M)
        if rec is None:
            continue
        rows.append({'kinase': k, 'group': group_map.get(k), 'method': name, **rec})
    df = pd.DataFrame(rows)

    def z(s):
        return (s - s.mean()) / s.std(ddof=0)
    comp = pd.concat([z(df[c]) for c in COMPOSITE_UP] + [-z(df['eff_positions'])], axis=1)
    df['spec_index'] = comp.mean(axis=1)
    print(f'  {name:8} {len(df)} kinases | spec_index range '
          f'{df.spec_index.min():.2f}..{df.spec_index.max():.2f}')
    return df


# ---------------------------------------------------------------- figures


def plot_bars(df, method):
    "Per-kinase spec_index bar, ordered/coloured by kinase group."
    plot_bar(df, value='spec_index', group='group', palette=group_color, figsize=(9, 3))
    plt.ylabel('Specificity index')
    plt.title(f'{method} — per-kinase sharpness (spec_index), by group')
    save_svg(FIG / f'spec_metrics_bar_{method.lower().replace("-", "_")}.svg')
    plt.close('all')


def plot_bygroup(long):
    "Mean spec_index per kinase group, grouped bars by method — which groups are sharp?"
    wide = long.pivot_table(index='group', columns='method', values='spec_index', aggfunc='mean')
    order = [g for g in group_color if g in wide.index]
    wide = wide.reindex(order)[[m for m in METHOD_ORDER if m in wide.columns]].reset_index()
    ax = plot_group_bar(wide, value_cols=[m for m in METHOD_ORDER if m in wide.columns],
                        group='group', order=order, figsize=(11, 4), rotation=25, fontsize=11,
                        palette='Set2', title='Mean sharpness (spec_index) per kinase group')
    ax.set_ylabel('mean spec_index')
    ax.axhline(0, color='0.7', lw=.6)
    save_svg(FIG / 'spec_metrics_bygroup.svg')
    plt.close('all')


def plot_rank_fig(df, method):
    "Ranked dot plot of per-kinase spec_index, high→low, coloured by group; extremes labelled."
    plot_rank(df.sort_values('spec_index', ascending=False), x='kinase', y='spec_index',
              hue='group', figsize=(7, 3), legend=False, palette=group_color, n_hi=12, n_lo=12)
    plt.xlabel('Rank (most → least specific)')
    plt.ylabel('Specificity index')
    plt.title(f'{method} — most vs least specific kinases')
    save_svg(FIG / 'spec_metrics_rank.svg')
    plt.close('all')


def plot_cross_method(long, group_map):
    "Do the methods agree on which kinases are sharp? Pairwise spec_index scatters."
    wide = long.pivot(index='kinase', columns='method', values='spec_index')
    wide['group'] = wide.index.map(group_map)
    pairs = [('PSPA', 'CDDM'), ('PSPA', 'MLP-attr'), ('CDDM', 'MLP-attr')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, (a, b) in zip(axes, pairs):
        if a not in wide or b not in wide:
            continue
        d = wide.dropna(subset=[a, b])
        for g, sub in d.groupby('group'):
            ax.scatter(sub[a], sub[b], s=13, color=group_color.get(g, '#999'), alpha=.7,
                       edgecolor='none', label=g)
        rho = spearmanr(d[a], d[b]).correlation
        ax.axhline(0, color='0.85', lw=.5)
        ax.axvline(0, color='0.85', lw=.5)
        ax.set_xlabel(f'{a} spec_index')
        ax.set_ylabel(f'{b} spec_index')
        ax.set_title(f'{a} vs {b}   ρ = {rho:.2f}  (n = {len(d)})', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1].legend(fontsize=6, ncol=2, frameon=False, loc='lower right')
    fig.suptitle('Do PSPA / CDDM / MLP-attr agree on which kinases are specific?', fontsize=12)
    save_svg(FIG / 'spec_metrics_cross_method.svg')
    plt.close('all')


def cddm_pspa_agreement():
    """Per-kinase CDDM-vs-PSPA agreement = AP@5 (does CDDM recover PSPA's 5 strongest determinants).

    Determinant-focused, not per-position Spearman: specificity lives in a few strong cells, so AP@5
    is the honest agreement metric. Reuses motif_16's implementation (same as motif_15).
    """
    from motif_16_compare_methods import average_precision, recover
    cddm = kdata.load('cddm')
    pspa = kdata.load('pspa')
    pspa = pspa[~pspa.index.str.contains('_TYR')]
    out = {k: average_precision(recover(cddm.loc[k]), recover(pspa.loc[k]))
           for k in sorted(set(cddm.index) & set(pspa.index))}
    return pd.Series(out, name='agreement').dropna()


def plot_agreement_vs_index(long, group_map):
    "CDDM-vs-PSPA agreement per kinase against its sharpness (spec_index) — the motif_15 idea, redrawn."
    sw = long.pivot(index='kinase', columns='method', values='spec_index')
    x = sw[['PSPA', 'CDDM']].mean(axis=1)              # kinase sharpness = mean of the two methods
    df = pd.DataFrame({'spec_index': x, 'agreement': cddm_pspa_agreement()}).dropna()
    df['group'] = df.index.map(group_map)
    rho = df.spec_index.corr(df.agreement, method='spearman')
    plt.figure(figsize=(5.2, 4.2))
    for g, sub in df.groupby('group'):
        plt.scatter(sub.spec_index, sub.agreement, s=16, color=group_color.get(g, '#999'),
                    alpha=.7, edgecolor='none', label=g)
    plt.xlabel('specificity (mean spec_index, PSPA & CDDM)')
    plt.ylabel("CDDM–PSPA agreement (AP@5: CDDM recovers PSPA's top-5)")
    plt.title(f'Do the methods agree more on specific kinases?  ρ = {rho:.2f} (n = {len(df)})', fontsize=10)
    plt.legend(fontsize=6, ncol=2, frameon=False)
    plt.tight_layout()
    save_svg(FIG / 'spec_agreement_vs_index.svg')
    plt.close('all')
    print(f'  agreement vs spec_index: Spearman rho = {rho:.2f} (n = {len(df)})')


def plot_shape(long, method):
    "Two angles on the sharpness 'shape', on one method, coloured by group."
    d = long[long.method == method]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for ax, (x, y, xl, yl) in zip(axes, [
            ('eff_positions', 'sharp_max', 'eff. # of positions (↓ peaky)', 'sharpest position (Gini)'),
            ('gini', 'kurtosis', 'cell concentration (Gini)', 'outlier heaviness (kurtosis)')]):
        for g, sub in d.groupby('group'):
            ax.scatter(sub[x], sub[y], s=14, color=group_color.get(g, '#999'), alpha=.7,
                       edgecolor='none', label=g)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1].legend(fontsize=6, ncol=2, frameon=False)
    fig.suptitle(f'Shape of specificity — {method} (each dot a kinase)', fontsize=12)
    save_svg(FIG / 'spec_metrics_shape.svg')
    plt.close('all')


def main():
    set_sns()
    info = kdata.load('kinase_info')
    group_map = info[info.pseudo == '0'].drop_duplicates('kinase').set_index('kinase')['group'].to_dict()

    pspa = kdata.load('pspa')
    pspa = pspa[~pspa.index.str.contains('_TYR')]      # drop dual-specificity duplicates
    sources = {'PSPA': pspa, 'CDDM': kdata.load('cddm')}
    if MLP_ATTR.exists():
        sources['MLP-attr'] = pd.read_parquet(MLP_ATTR)
    else:
        print(f'skip MLP-attr: {MLP_ATTR} not found (run motif_17)')

    print('building per-kinase sharpness metrics:')
    long = pd.concat([build_method(n, df, group_map) for n, df in sources.items()], ignore_index=True)

    out = OUT / 'specificity_metrics.parquet'
    long.to_parquet(out, index=False)
    long.to_csv(OUT / 'specificity_metrics.csv', index=False)
    print('wrote', out, long.shape)

    for m in sources:
        plot_bars(long[long.method == m], m)
    plot_bygroup(long)
    plot_rank_fig(long[long.method == 'PSPA'], 'PSPA')
    plot_cross_method(long, group_map)
    plot_shape(long, 'PSPA')
    plot_agreement_vs_index(long, group_map)
    print('figures written to', FIG)


if __name__ == '__main__':
    main()
