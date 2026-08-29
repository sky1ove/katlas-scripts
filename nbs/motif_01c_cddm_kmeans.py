"""motif_01c · CDDM sub-motifs by k-means clustering of each kinase's sites (exploratory).

For each kinase, cluster its (deduplicated) substrate sites by the one-hot of the ±5 flanks —
the centre residue is dropped so clusters capture flank sub-motifs rather than just the S/T
acceptor — and pick k in 2..K_MAX by the highest Calinski-Harabasz score. This is a **descriptive**
stratification: Calinski-Harabasz always returns a best k ≥ 2, so every kinase is split. The flank
sub-structure is genuinely weak (silhouette ~0.03; the BIC mixture in raw_scripts/cddm_mixture_bic.py leaves *all*
kinases unsplit), so treat these sub-motifs as exploratory, not statistically-validated subtypes.
k is capped so every cluster *can* hold ≥ MIN_CLUSTER sites; each cluster with ≥ MIN_CLUSTER sites
gets a frequency PSSM over the full ±20 window (same schema as `cddm`; frequency, not log-odds).

Same site set as the CDDM build (motif_01): `ks_dataset(thr=40)`, case-insensitive dedup, kinases
with ≥ MIN_SITES sites. Clustering keeps the sequence case-preserving, so lower-case phospho-priming
residues (s/t/y) at flanking positions are their own features; the emitted PSSMs are likewise
case-preserving, as in `cddm`.

KMeans uses `n_init="auto"` (10 k-means++ restarts, best inertia) with a fixed seed, so no manual
seed sweep is needed and the result is reproducible.

Inputs   kdata.ks_dataset(thr=40)
Outputs  out/cddm_kmeans.parquet — one row per (kinase, cluster): freq PSSM (±20) + n_sites,
         best_k, ch

Run:  python nbs/motif_01c_cddm_kmeans.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from paths import OUT
from sklearn.metrics import calinski_harabasz_score

import kdata
from katlas.pssm import flatten_pssm, get_prob
from kprot.onehot import filter_range_columns, onehot_encode, run_kmeans

MIN_SITES = 40       # kinases to consider (matches the CDDM build in motif_01)
WINDOW = 5           # cluster on the ±WINDOW flanks
MIN_CLUSTER = 15     # a cluster needs this many sites for a reliable frequency PSSM
K_MAX = 10           # sweep k in 2..K_MAX
SEED = 42


def cluster_features(sites):
    "One-hot of the ±WINDOW flanks, centre residue dropped, case-preserving (lower-case s/t/y = phospho-priming)."
    oh = onehot_encode(list(sites))                         # case-preserving: s/t/y kept distinct from S/T/Y
    oh = filter_range_columns(oh, -WINDOW, WINDOW)
    return oh.loc[:, oh.columns.str[:-1].astype(int) != 0]   # drop the centre (position 0)


def choose_k(X):
    "(k, ch, labels): best k in 2..K_MAX by Calinski-Harabasz (always splits; k capped by MIN_CLUSTER)."
    k_max = min(K_MAX, len(X) // MIN_CLUSTER)
    if k_max < 2:                                            # too few sites to split
        return 1, float('nan'), np.zeros(len(X), int)
    labels_by_k, ch = {}, {}
    for k in range(2, k_max + 1):
        labels_by_k[k] = run_kmeans(X, n=k, seed=SEED)
        ch[k] = calinski_harabasz_score(X, labels_by_k[k])
    best_k = max(ch, key=ch.get)
    return best_k, ch[best_k], labels_by_k[best_k]


def kinase_clusters(g):
    "List of (cluster_id, cluster_rows, k, ch) for one kinase, keeping clusters ≥ MIN_CLUSTER."
    k, ch, labels = choose_k(cluster_features(g.site_seq.values))
    return [(c, g[labels == c], k, ch) for c in sorted(set(labels)) if (labels == c).sum() >= MIN_CLUSTER]


def main():
    data = kdata.ks_dataset(thr=40)
    data = data.assign(seq_upper=data.site_seq.str.upper()).drop_duplicates(['kinase_protein', 'seq_upper'])
    keep = data.kinase_protein.value_counts()
    subset = data[data.kinase_protein.isin(keep[keep >= MIN_SITES].index)]
    print(f'{subset.kinase_protein.nunique()} kinases, {len(subset)} deduped sites')

    rows, dropped = [], 0
    for kinase, g in subset.groupby('kinase_protein'):
        g = g.reset_index(drop=True)
        clusters = kinase_clusters(g)
        dropped += g.shape[0] - sum(len(gc) for _, gc, _, _ in clusters)
        for cid, gc, k, ch in clusters:
            rec = {'kinase': kinase, 'cluster': cid, 'n_sites': len(gc), 'best_k': k, 'ch': ch}
            rec.update(flatten_pssm(get_prob(gc, 'site_seq')))   # freq PSSM (dict) over full ±20
            rows.append(rec)

    out = pd.DataFrame(rows)
    meta = ['kinase', 'cluster', 'n_sites', 'best_k', 'ch']
    out = out[meta + [c for c in out.columns if c not in meta]]
    path = OUT / 'cddm_kmeans.parquet'
    out.to_parquet(path, index=False)

    per_kinase_k = out.groupby('kinase').best_k.first()
    print(f'wrote {path} {out.shape} (descriptive / exploratory — every kinase split)')
    print(f'  best_k distribution: {dict(per_kinase_k.value_counts().sort_index())} | '
          f'sites in dropped sub-MIN_CLUSTER clusters: {dropped}')


if __name__ == '__main__':
    main()
