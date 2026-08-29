"""kd_02a · One-hot features for each active kinase domain.

The aligned residue at every well-populated alignment column, one-hot encoded into a per-domain
feature vector, plus a PCA reduction so the one-hot (thousands of columns) is comparable in width to
the ~1k-dimensional T5 / ESM embeddings (kd_02b) — this is what lets kd_04b ask which
representation best predicts substrate specificity on an even footing. Restricted to the 4,209
catalytically active domains (`active_D1_D2`).

Only alignment columns where some residue reaches FREQ_CUT are encoded — an almost-all-gap column
carries no signal and would just add noise and width.

Inputs   out/kd_align.parquet, out/kd_motif_labeled.parquet (kd_01b)
Outputs  out/kd_feat_onehot.parquet, out/kd_feat_onehot_pca.parquet

Run:  python nbs/kd_02a_onehot.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from paths import OUT

from katlas.utils import get_aln_freq
from kplot.scatter import reduce_feature

FREQ_CUT = 0.05     # keep an alignment column if some residue reaches this frequency
PCA_DIM = 1000      # match the ~1k dimensionality of the T5 / ESM embeddings


def active_alignment():
    "The alignment restricted to catalytically active domains (active_D1_D2)."
    align = pd.read_parquet(OUT / 'kd_align.parquet')
    align.columns = align.columns.astype(int)

    kd = pd.read_parquet(OUT / 'kd_motif_labeled.parquet')
    active = kd.loc[kd.active_D1_D2.astype(bool), 'kd_ID']
    align = align.loc[align.index.intersection(active)]
    print('active domains:', align.shape)
    return align


def informative_columns(align):
    "Alignment columns where some residue reaches FREQ_CUT (gaps excluded)."
    freq = get_aln_freq(align).iloc[1:, :]          # drop the '-' row
    best = freq.max()
    cols = sorted(best[best > FREQ_CUT].index)
    print(f'informative columns: {len(cols)} / {align.shape[1]} (residue above {FREQ_CUT:.0%})')
    return cols


def onehot_encode(align):
    "One-hot each alignment column; column names are `{position}_{residue}`."
    from sklearn.preprocessing import OneHotEncoder
    out = []
    for col in align.columns:
        enc = OneHotEncoder(sparse_output=False, dtype=int, handle_unknown='ignore')
        encoded = enc.fit_transform(align[[col]])
        sub = pd.DataFrame(encoded, index=align.index,
                           columns=[f'{col}_{aa}' for aa in enc.categories_[0]])
        out.append(sub)
    return pd.concat(out, axis=1)


def main():
    align = active_alignment()
    align = align[informative_columns(align)]

    onehot = onehot_encode(align)
    print('onehot:', onehot.shape)
    onehot.to_parquet(OUT / 'kd_feat_onehot.parquet')

    onehot_pca = reduce_feature(onehot, n=PCA_DIM)
    print('onehot_pca:', onehot_pca.shape)
    onehot_pca.to_parquet(OUT / 'kd_feat_onehot_pca.parquet')

    print('\nwrote:')
    for n in ['onehot', 'onehot_pca']:
        p = OUT / f'kd_feat_{n}.parquet'
        print(f'  {p} {pd.read_parquet(p).shape}')


if __name__ == '__main__':
    main()
