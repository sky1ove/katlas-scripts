"""motif_01d · CDDM plot example — one kinase's CDDM and its stratifications (single-kinase inspection).

Stacks the kinase's overall CDDM logo and each of its stratified sub-motif logos into one figure.
`DATASET` selects the stratification:
  cddm      overall CDDM only            (kdata)
  acceptor  per phospho-acceptor S/T/Y   (motif_01b → out/cddm_acceptor.parquet)
  kmeans    per k-means sub-motif        (motif_01c → out/cddm_kmeans.parquet)

`plot_kinase(kinase, dataset)` returns the matplotlib figure and is imported by web_01 for
batch export; run as a script to save one example.

The overall CDDM ('cddm') is drawn as a logo + heatmap; the stratifications stack the overall
logo and each sub-motif logo.

Inputs   kdata.load('cddm'); out/cddm_acceptor.parquet, out/cddm_kmeans.parquet
Outputs  fig/<kinase>_<dataset>.svg   (dataset ∈ cddm / acceptor / kmeans)

Run:  python nbs/motif_01d_cddm_plot_example.py [KINASE] [DATASET]     # no DATASET -> all three; default kinase CDK7
"""
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from paths import FIG, OUT

import kdata
from katlas.lo import plot_logo_heatmap_LO
from katlas.plot import plot_logo, plot_logo_heatmap
from katlas.pssm import recover_pssm
from kplot.utils import save_svg

#: dataset -> the column that labels each stratum (None = a single overall matrix, not stacked)
STRAT = {'cddm': None, 'cddm_lo': None, 'acceptor': 'acceptor', 'kmeans': 'cluster'}
META = ['kinase', 'cluster', 'acceptor', 'n_sites', 'best_k', 'ch', 'weight', 'bic']


@lru_cache
def _site_counts():
    "Per-kinase deduplicated site count (the overall `n`, same filter as the CDDM build)."
    d = kdata.ks_dataset(thr=40)
    d = d.assign(u=d.site_seq.str.upper()).drop_duplicates(['kinase_protein', 'u'])
    return d.kinase_protein.value_counts().to_dict()


def _kmeans_forced(kinase, k):
    "Re-cluster one kinase's sites into exactly `k` k-means sub-motifs (motif_01c features, same seed); "
    "returns [(label, freq PSSM)] per cluster. Use when the stored cddm_kmeans.parquet (auto-k) is not "
    "the k you want to show; deterministic, so k = the stored auto-k reproduces the stored split."
    from motif_01c_cddm_kmeans import SEED, cluster_features
    from katlas.pssm import flatten_pssm, get_prob
    from kprot.onehot import run_kmeans
    d = kdata.ks_dataset(thr=40)
    d = d.assign(u=d.site_seq.str.upper()).drop_duplicates(['kinase_protein', 'u'])
    g = d[d.kinase_protein == kinase].reset_index(drop=True)
    labels = run_kmeans(cluster_features(g.site_seq.values), n=k, seed=SEED)
    out = []
    for c in sorted(set(labels)):
        gc = g[labels == c]
        out.append((f'cluster {c} (n={len(gc)})',
                    recover_pssm(pd.Series(flatten_pssm(get_prob(gc, 'site_seq'))))))
    return out


def _panels(kinase, dataset, k=None):
    "List of (label, 2D PSSM): the overall CDDM plus each stratified sub-motif for the kinase. For "
    "dataset='kmeans', `k` re-clusters into exactly k sub-motifs instead of reading the stored auto-k."
    panels = [(f'overall CDDM (n={_site_counts().get(kinase, "?")})',
               recover_pssm(kdata.load('cddm').loc[kinase]))]
    key = STRAT[dataset]
    if key is None:
        return panels
    if dataset == 'kmeans' and k is not None:
        return panels + _kmeans_forced(kinase, k)                  # forced cluster count, re-clustered
    df = pd.read_parquet(OUT / f'cddm_{dataset}.parquet')
    rows = df[df.kinase == kinase]
    if rows.empty:
        raise SystemExit(f'{kinase} not in cddm_{dataset}.parquet')
    pssm_cols = [c for c in df.columns if c not in META]
    for _, r in rows.iterrows():
        panels.append((f'{key} {r[key]} (n={int(r.n_sites)})',
                       recover_pssm(r[pssm_cols].astype(float))))   # row is object-dtype -> cast
    return panels


def plot_kinase(kinase, dataset='kmeans', window=None, cddm_figsize=(13, 6), figsize=None,
                hspace=0.12, suptitle=True, title=None, bold_cluster=None, k=None):
    "Figure: the overall CDDM as a logo+heatmap ('cddm', optionally windowed to ±`window`), or overall + stratified sub-motif logos."
    if dataset not in STRAT:
        raise SystemExit(f'DATASET must be one of {list(STRAT)}')
    if dataset in ('cddm', 'cddm_lo'):
        is_lo = dataset == 'cddm_lo'
        m = recover_pssm(kdata.load('cddm_LO' if is_lo else 'cddm').loc[kinase])
        if window is not None:
            m = m.loc[:, [c for c in m.columns if abs(int(c)) <= window]]
        label = 'CDDM log-odds' if is_lo else 'CDDM'
        title = f'{kinase} - {label} (n={_site_counts().get(kinase, "?")})'
        plotter = plot_logo_heatmap_LO if is_lo else plot_logo_heatmap
        plotter(m, title=title, figsize=cddm_figsize, square=True, hspace=hspace)
        return plt.gcf()
    panels = _panels(kinase, dataset, k=k)
    fig, axes = plt.subplots(len(panels), 1, figsize=figsize or (10, 1.5 * len(panels)))
    axes = [axes] if len(panels) == 1 else axes
    for ax, (label, pssm) in zip(axes, panels):
        if window is not None:                                       # limit the logo to +/- window
            pssm = pssm.loc[:, [c for c in pssm.columns if abs(int(c)) <= window]]
        plot_logo(pssm, title=label, ax=ax)
        bold = bold_cluster is not None and label.startswith(f'cluster {bold_cluster} ')  # winning sub-motif
        ax.set_title(label, fontsize=7, pad=2, fontweight='bold' if bold else 'normal')   # spec: 7 pt
        ax.tick_params(length=2, pad=1.5, labelsize=7)               # short ticks; 7 pt overrides plot_logo's
        if window is not None:                                       # narrow-font 10 pt x-ticks. When windowed
            for t in ax.get_xticklabels():                           # (e.g. +/-5) there is room for the normal
                t.set_fontfamily('sans-serif')                       # font; the condensed one is only for the
                                                                     # cramped full +/-20 axis (plot_logo_raw)
        ax.set_ylabel(ax.get_ylabel(), labelpad=1)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))     # 1-decimal y ticks so widths line up
    fig.tight_layout(h_pad=0.3)
    if suptitle:                                                     # reserve a fixed absolute band (in) for
        head = min(0.30, 0.35 / fig.get_figheight())                # the title, so changing the figure
        fig.subplots_adjust(top=1 - head)                           # height/ratio does not move it
        pos = axes[0].get_position()                                # center over the logo axes, not the whole
        cx = pos.x0 + pos.width / 2                                 # figure, so the y-label does not pull it left
        fig.suptitle(title or f'{kinase} - {dataset}', x=cx, y=1 - head * 0.4, fontsize=7, fontweight='bold')
    return fig


def plot_kmeans_stratification(kinase, window=5, figsize=None, suptitle=True, title=None,
                               bold_cluster=None, k=None):
    """One kinase's k-means sub-motifs as a stacked logo figure (paper styling), the dedicated entry
    point for the stratification panels. Stacks the overall CDDM logo and each k-means cluster logo,
    windowed to +/-`window`; 7 pt fonts, short ticks, normal (non-condensed) x-axis font, one-decimal
    y ticks, and a bold title centered over the logos. Size via `figsize` (e.g.
    `kplot.utils.paper_panel(1/3, ratio=1.05)`); `title` overrides the default suptitle (e.g. the kinase
    with its group); `bold_cluster` bolds that cluster's per-logo title (the PSPA-matching winner);
    `k` forces exactly k clusters (re-clustered) instead of the stored auto-k. Reads out/cddm_kmeans.parquet
    (motif_01c) when `k` is None.
    """
    return plot_kinase(kinase, dataset='kmeans', window=window, figsize=figsize, suptitle=suptitle,
                       title=title, bold_cluster=bold_cluster, k=k)


def main():
    kinase = sys.argv[1] if len(sys.argv) > 1 else 'CDK7'
    datasets = [sys.argv[2]] if len(sys.argv) > 2 else list(STRAT)
    for dataset in datasets:
        fig = plot_kinase(kinase, dataset)
        out = FIG / f'{kinase}_{dataset}.svg'
        save_svg(out)
        plt.close(fig)
        print(f'wrote {out}')
    if 'cddm' in datasets:                       # also a ±5 zoom of the overall CDDM core
        fig = plot_kinase(kinase, 'cddm', window=5)
        out = FIG / f'{kinase}_cddm_w5.svg'
        save_svg(out)
        plt.close(fig)
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
