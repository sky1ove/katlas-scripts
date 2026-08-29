"""motif_07 · Validate the specificity dendrograms against the kinase taxonomy.

For each specificity method — CDDM (frequency), PSPA, and the CDDM MLP-attribution — the per-kinase
clustering vector built by motif_06 (row-standardized, so Euclidean + Ward clusters on the 1 - Pearson
geometry) is clustered and scored against the CORAL kinase group / family / subfamily labels:

  silhouette   how well the distance separates the true classes — cut-free (uses the labels, not a cut)
  ARI          adjusted Rand index of the tree cut to the number of classes vs the labels
  AMI          adjusted mutual information of the same cut (NMI without the many-cluster inflation)

Two kinase sets. `overlap_PSPA_CDDM` scores every method on the kinases PSPA and CDDM share (identical
set); `own_n` scores each method on its own full kinase set.

Each is computed twice — **with position 0** (the acceptor, per the clustering rule) and **flank-only**
(position 0 excluded). Position 0 is the S/T-vs-Y acceptor, which trivially recovers the TK group, so
including it inflates the group-level scores; and it is encoded unevenly across methods — measured for
CDDM, **assay-assigned** for PSPA (separate S/T and Y arrays), and a hard **is-TK `in_y_branch` label**
for MLP-attr (its two-branch build has no clean acceptor). The flank-only variant removes that split
(and the in_y_branch) so the scores reflect flank specificity alone — the honest cross-method check.

Inputs   motif_06 cluster vectors (kdata: cddm, pspa_scale; out/mlp_attr_pssm_full.parquet),
         kinase_info (group / family / subfamily labels)
Outputs  out/cluster_validation.xlsx  (sheets: overlap_PSPA_CDDM, own_n, overlap_flank_only, own_n_flank_only)
         fig/cluster_validation_{overlap,own}{,_flank}.svg (grouped bars)

Run:  python nbs/motif_07_cluster_validation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from motif_06_pssm_hierarchy import cddm_cluster_vector, mlp_attr_cluster_matrix, pspa_cluster_vector
from motif_util import kinase_group_map
from paths import FIG, OUT
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score

import kdata
from kplot.utils import paper_panel, save_svg

LEVELS = ['group', 'family', 'subfamily']
METHOD_ORDER = ['PSPA', 'CDDM', 'CDDM MLP-attr']       # PSPA first — easier to read the bars against
METHOD_COLORS = {'CDDM': '#c0392b', 'PSPA': '#2166ac', 'CDDM MLP-attr': '#2e8b57'}
METRICS = ['silhouette', 'ARI', 'AMI']


#: legend labels: drop the "CDDM " prefix from MLP-attr, matching the rest of the paper
METHOD_LABEL = {'PSPA': 'PSPA', 'CDDM': 'CDDM', 'CDDM MLP-attr': 'MLP-attr'}


def plot_silhouette(df, out, frac=1 / 3, ratio=1.15):
    "Standalone silhouette (cut-free) panel: taxonomy level on x, method as color; paper-sized, on-spec fonts."
    methods = [m for m in METHOD_ORDER if m in set(df.method)]
    fig, ax = plt.subplots(figsize=paper_panel(frac, ratio=ratio))
    x = np.arange(len(LEVELS)); w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        sub = df[df.method == m].set_index('level').reindex(LEVELS)['silhouette']
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, sub.to_numpy(), w,
               color=METHOD_COLORS[m], label=METHOD_LABEL[m])
    ax.set_xticks(x); ax.set_xticklabels(LEVELS)
    ax.set_title('silhouette (cut-free)', fontsize=8)
    ax.axhline(0, color='0.7', lw=0.5)
    ax.set_ylabel('score (higher = recovers the taxonomy better)', labelpad=2)
    ax.tick_params(length=2, pad=1.5)
    ax.legend(fontsize=6, frameon=False, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    save_svg(out); plt.close(fig)
    print('  wrote', out)


def plot_validation(df, tag, out):
    "Grouped bars: one panel per metric, taxonomy level on x, method as the colour."
    methods = [m for m in METHOD_ORDER if m in set(df.method)]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(12, 3.6), constrained_layout=True)
    x = np.arange(len(LEVELS))
    w = 0.8 / len(methods)
    for ax, metric in zip(axes, METRICS):
        for i, m in enumerate(methods):
            sub = df[df.method == m].set_index('level').reindex(LEVELS)[metric]
            ax.bar(x + (i - (len(methods) - 1) / 2) * w, sub.to_numpy(), w,
                   color=METHOD_COLORS[m], label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(LEVELS)
        ax.set_title(f'{metric}  ({"cut-free" if metric == "silhouette" else "tree cut"})', fontsize=10)
        ax.axhline(0, color='0.7', lw=0.5)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('score (higher = recovers the taxonomy better)')
    axes[-1].legend(fontsize=8.5, frameon=False, loc='upper right')
    fig.suptitle(f'Do the specificity clusters recapitulate the CORAL taxonomy? — {tag}', fontsize=12)
    save_svg(out)
    plt.close('all')
    print('  wrote', out)


def method_vectors(flank_only=False):
    """Each method's row-standardized clustering vectors as a DataFrame keyed by kinase.

    `flank_only=True` drops position 0 (the acceptor) — and the MLP-attr in_y_branch is-TK indicator —
    so the scores measure flank specificity alone, not the trivial S/T-vs-Y split (which is measured
    for CDDM, assay-assigned for PSPA, and a hard label for MLP-attr).
    """
    incl0 = not flank_only
    group_of = kinase_group_map(kdata.load('kinase_info'))
    X, attr = mlp_attr_cluster_matrix(group_of, incl0=incl0, add_iyb=incl0)
    return {'CDDM': cddm_cluster_vector(incl0=incl0),
            'PSPA': pspa_cluster_vector(incl0=incl0),
            'CDDM MLP-attr': pd.DataFrame(X, index=attr.index)}


def score(vec, restrict, ki, method):
    "Cluster `vec` (optionally restricted to `restrict`) and score each taxonomy level."
    idx = [k for k in vec.index if restrict is None or k in restrict]
    D = pdist(vec.loc[idx].to_numpy(), metric='euclidean')
    Z = linkage(D, method='ward')
    Dsq = squareform(D)
    rows = []
    for level in LEVELS:
        lab = ki[level].reindex(idx)
        keep = lab.notna().to_numpy()
        y = lab[keep].astype(str).to_numpy()
        K = len(set(y))
        cl = fcluster(Z, K, criterion='maxclust')[keep]
        rows.append({'method': method, 'level': level, 'n': int(keep.sum()), 'K': K,
                     'silhouette': round(silhouette_score(Dsq[keep][:, keep], y, metric='precomputed'), 3),
                     'ARI': round(adjusted_rand_score(y, cl), 3),
                     'AMI': round(adjusted_mutual_info_score(y, cl), 3)})
    return rows


def table(vecs, restrict, ki):
    df = pd.DataFrame(r for m, v in vecs.items() for r in score(v, restrict, ki, m))
    df['level'] = pd.Categorical(df['level'], categories=LEVELS, ordered=True)
    return df.sort_values(['level', 'method']).reset_index(drop=True)


def main():
    ki = kdata.load('kinase_info').drop_duplicates('kinase').set_index('kinase')

    # two variants: (a) with position 0 (the acceptor, per the clustering rule) and (b) flank-only.
    # Position 0 trivially recovers the TK group (S/T vs Y), so it inflates group-level scores and is
    # encoded unevenly across methods (measured for CDDM, assay-assigned for PSPA, a hard is-TK label
    # for MLP-attr) — the flank-only variant is the honest cross-method robustness check.
    vecs = method_vectors(flank_only=False)
    vecs_f = method_vectors(flank_only=True)
    overlap = set(vecs['PSPA'].index) & set(vecs['CDDM'].index)
    print(f'PSPA ∩ CDDM = {len(overlap)} kinases')

    shared = table(vecs, overlap, ki)
    own = table(vecs, None, ki)
    shared_f = table(vecs_f, overlap, ki)
    own_f = table(vecs_f, None, ki)
    print('\n== overlap (PSPA ∩ CDDM), incl. position 0 ==\n', shared.to_string(index=False))
    print('\n== overlap (PSPA ∩ CDDM), flank-only ==\n', shared_f.to_string(index=False))

    out = OUT / 'cluster_validation.xlsx'
    with pd.ExcelWriter(out) as xl:
        shared.to_excel(xl, sheet_name='overlap_PSPA_CDDM', index=False)
        own.to_excel(xl, sheet_name='own_n', index=False)
        shared_f.to_excel(xl, sheet_name='overlap_flank_only', index=False)
        own_f.to_excel(xl, sheet_name='own_n_flank_only', index=False)
    print('\nwrote', out)

    plot_validation(shared, f'PSPA ∩ CDDM shared (n={len(overlap)}) · incl. position 0',
                    FIG / 'cluster_validation_overlap.svg')
    plot_validation(shared_f, f'PSPA ∩ CDDM shared (n={len(overlap)}) · flank-only (position 0 excluded)',
                    FIG / 'cluster_validation_overlap_flank.svg')
    plot_validation(own, "each method's own set · incl. position 0", FIG / 'cluster_validation_own.svg')
    plot_validation(own_f, "each method's own set · flank-only", FIG / 'cluster_validation_own_flank.svg')

    plot_silhouette(shared,   FIG / 'silhouette_overlap.svg')          # standalone panels, frac 1/3
    plot_silhouette(shared_f, FIG / 'silhouette_overlap_flank.svg')
    plot_silhouette(own,      FIG / 'silhouette_own.svg')
    plot_silhouette(own_f,    FIG / 'silhouette_own_flank.svg')


if __name__ == '__main__':
    main()
