"""motif_06 · Hierarchical clustering of kinases by substrate specificity.

Three dendrograms - from the CDDM frequency PSSMs, the scaled PSPA matrices, and the CDDM
MLP-attribution PSSMs - grouping kinases whose substrate preferences look alike. Branches and
leaf labels are coloured by kinase group (a branch mixing groups goes gray), and each leaf is
labelled with the kinase's consensus motif, laid out so the central `s/t/y*` lines up down the figure.

Comparing the two answers whether specificity measured from observed substrates (CDDM) and
from peptide arrays (PSPA) organise the kinome the same way.

Distance: both methods cluster by preference *pattern* (correlation), not magnitude. Each kinase's
flattened PSSM is standardized to mean 0 / std 1 over its own cells, which makes Euclidean distance
monotonic in 1 - Pearson (‖ẑ_a − ẑ_b‖² = 2p(1 − r)); **Euclidean + Ward** on the standardized vectors
then clusters on the correlation geometry while keeping Ward's balanced (cut-able) clusters. PSPA
runs on `pspa_scale`, drops the pS/pT duplicate `t` (the array reports pS == pT), and keeps the full
matrix. CDDM runs on the frequency (not log-odds, which is noisy for low-site-count kinases) over the
±5 flank (the far flank is mostly background), with s/t/y kept distinct. The central acceptor (the
strongest specificity signal, S/T vs Y) is kept in both. This recapitulates the kinase
groups/subfamilies best — see the validation below.

The CDDM MLP-attribution matrix (`out/mlp_attr_pssm_full.parquet`, from motif_17) is a
signed ±5 attribution PSSM; it is standardized and clustered the same way, with its consensus-motif
labels read from the positive attributions. Because it concatenates two separately-trained branches
(a tyrosine model for TK kinases, a serine/threonine model for the rest) it does not cleanly encode
the acceptor, so an `in_y_branch` indicator (weighted by the max z-distance) is appended, making the
tree split Tyr vs S/T at the top and cluster by attribution within. Skipped if the file is absent.

Inputs   kdata: cddm, pspa_scale, kinase_info, ks_dataset(thr=40) for the label counts;
         out/mlp_attr_pssm_full.parquet (motif_17)
Outputs  fig/CDDM_hierarchy.svg, fig/PSPA_hierarchy.svg, fig/MLP_attr_hierarchy.svg

Run:  python nbs/motif_06_pssm_hierarchy.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from motif_util import (add_group_legend, get_aligned_labels, get_group_colors,
                        kinase_group_map, style_yticklabels)
from paths import FIG, OUT
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

import kdata
from kplot.hierarchical import get_Z, plot_dendrogram
from kplot.utils import save_svg

MOTIF_THR = 0.2     # min frequency for a residue to enter the consensus motif label (freq PSSMs)
ATTR_THR = 0.5      # same, for the signed MLP-attribution matrix (attribution units, not frequency)
MLP_ATTR = OUT / 'mlp_attr_pssm_full.parquet'

LEAF_COLOR = '#174ea6'   # dark blue for all leaf text (kinase name + consensus motif); branches stay group-coloured
LEAF_FONT = 11           # larger leaf labels for on-page legibility (default dendrogram font is 7)


def zscore_rows(v):
    "Standardize each kinase's flattened vector to mean 0 / std 1 — Euclidean+Ward then = 1-Pearson geometry."
    return v.sub(v.mean(axis=1), axis=0).div(v.std(axis=1).replace(0, 1), axis=0)


def flank_cols(cols, incl0=True):
    "Column names for the ±5 flank; when `incl0` is False the position-0 (acceptor) columns are dropped."
    out = []
    for c in cols:
        m = re.fullmatch(r'(-?\d+)([A-Za-z])', str(c))
        if m and abs(int(m.group(1))) <= 5 and (incl0 or int(m.group(1)) != 0):
            out.append(c)
    return out


def pspa_cluster_vector(incl0=True):
    """Per-kinase PSPA vector, row-standardized so Ward clusters on the 1 - Pearson geometry.

    PSPA is an enrichment measure, so kinases should be grouped by their preference *pattern*, not
    its magnitude — i.e. by correlation. Row-standardizing each kinase's flattened vector (z-score)
    makes Euclidean distance monotonic in 1 - Pearson (‖ẑ_a − ẑ_b‖² = 2p(1 − r)), so the default
    Euclidean + Ward then clusters on the correlation geometry *and* keeps Ward's balanced clusters —
    which recapitulate the kinase groups/subfamilies best (ARI/AMI beat both plain Euclidean/Ward and
    1 - Pearson/average; the latter's unbalanced trees cut poorly). Runs on `pspa_scale` (≥0, no
    missing cells; the log-enrichment form is too sparse — log of 0 — to use). The pS/pT duplicate `t`
    is dropped (the array reports pS == pT) to avoid double-weighting it. The central acceptor
    (position 0) is kept by default (strongest signal, S/T vs Y) but note it is **assay-assigned**
    here (separate S/T and Y arrays); `incl0=False` drops it for the flank-only robustness check.
    """
    v = kdata.load('pspa_scale').dropna(axis=1)
    v = v[[c for c in flank_cols(v.columns, incl0) if not re.fullmatch(r'-?\d+t', str(c))]]
    return zscore_rows(v)


def cddm_cluster_vector(incl0=True):
    """Per-kinase CDDM frequency vector for clustering, row-standardized like PSPA.

    Same rationale as PSPA (cluster by preference pattern → correlation → z-score + Ward), on the
    ±5 flank (the far flank is mostly background and only adds noise). CDDM keeps s/t/y as distinct
    phospho-primed residues (unlike PSPA, whose pS == pT) and keeps the central acceptor by default.
    Frequency is used rather than log-odds, which is noisy for the low-site-count kinases.
    `incl0=False` drops the position-0 acceptor columns (flank-only robustness check).
    """
    v = kdata.load('cddm')
    return zscore_rows(v[flank_cols(v.columns, incl0)])


def cddm_counts():
    "Sites per kinase behind the CDDM, using the same dedup as motif_01 - for the leaf labels."
    ks = kdata.ks_dataset(thr=40)
    ks = ks.assign(seq_upper=ks['site_seq'].str.upper())
    ks = ks.drop_duplicates(['kinase_protein', 'seq_upper'])
    return ks['kinase_protein'].value_counts()


def mlp_attr_cluster_matrix(group_of, incl0=True, add_iyb=True):
    """Standardized MLP-attr vectors + a weighted in_y_branch column. Returns (matrix, raw attr).

    The attribution concatenates two separately-trained branches (Tyr for TK kinases, S/T for the
    rest), so it doesn't cleanly encode the acceptor; an in_y_branch column weighted by the max
    z-distance (√2p) makes the tree split Tyr/S-T at the top and cluster by attribution within.
    `in_y_branch` is a hard is-TK indicator — it stands in for the missing acceptor — so for the
    flank-only robustness check both it and the position-0 cells are dropped (`incl0=add_iyb=False`).
    """
    attr = pd.read_parquet(MLP_ATTR)                        # signed ±5 attribution PSSM, s/t/y distinct
    z = zscore_rows(attr[flank_cols(attr.columns, incl0)])
    if not add_iyb:
        return z.to_numpy(), attr
    is_tyr = np.array([group_of(k) == 'TK' for k in attr.index], dtype=float)
    return np.column_stack([z.to_numpy(), is_tyr * np.sqrt(2 * z.shape[1])]), attr


def plot_hierarchy(pssms, group_of, out, dense=5, count_map=None, Z=None, thr=MOTIF_THR):
    "Dendrogram for `pssms`; pass a precomputed `Z` to use a custom distance/linkage (else Euclidean+Ward)."
    if Z is None:
        Z = get_Z(pssms)
    labels = get_aligned_labels(pssms, count_map=count_map, thr=thr)
    link_color_func, _ = get_group_colors(Z, pssms.index, group_of, labels=labels)
    label_colors = {lbl: LEAF_COLOR for lbl in labels}   # uniform dark-blue leaf text; branches stay group-coloured

    plot_dendrogram(Z, dense=dense, labels=labels, link_color_func=link_color_func, line_width=1.5)
    style_yticklabels(label_colors, monospace=True)
    for lbl in plt.gca().get_ymajorticklabels():          # bigger leaf labels than the dendrogram default
        lbl.set_fontsize(LEAF_FONT)
    add_group_legend(sorted({group_of(k) for k in pssms.index if group_of(k)}))
    save_svg(out)
    plt.close('all')
    print('  wrote', out)


def main():
    group_of = kinase_group_map(kdata.load('kinase_info'))

    print('== CDDM ==')
    cddm = kdata.load('cddm')                               # full matrix, for the consensus-motif labels
    vec = cddm_cluster_vector()                             # ±5, row-standardized (correlation geometry)
    print('  labels', cddm.shape, '| cluster vector', vec.shape)
    Z = linkage(pdist(vec.to_numpy(), metric='euclidean'), method='ward')   # Ward on the 1 - Pearson geometry
    plot_hierarchy(cddm, group_of, FIG / 'CDDM_hierarchy.svg', count_map=cddm_counts(), Z=Z)

    print('== PSPA ==')
    scale = kdata.load('pspa_scale').dropna(axis=1)         # full matrix, for the consensus-motif labels
    vec = pspa_cluster_vector()                             # row-standardized (correlation geometry)
    print('  labels', scale.shape, '| cluster vector', vec.shape)
    Z = linkage(pdist(vec.to_numpy(), metric='euclidean'), method='ward')   # Ward on the 1 - Pearson geometry
    plot_hierarchy(scale, group_of, FIG / 'PSPA_hierarchy.svg', Z=Z)

    if MLP_ATTR.exists():                                   # from motif_17
        print('== CDDM MLP-attr ==')
        vec, attr = mlp_attr_cluster_matrix(group_of)
        print('  labels', attr.shape, '| cluster vector', vec.shape)
        Z = linkage(pdist(vec, metric='euclidean'), method='ward')
        plot_hierarchy(attr, group_of, FIG / 'MLP_attr_hierarchy.svg', Z=Z, thr=ATTR_THR)
    else:
        print(f'skip MLP-attr: {MLP_ATTR} not found')


if __name__ == '__main__':
    main()
