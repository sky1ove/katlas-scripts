"""kd_04c · Deep-learning model (MLP) on the distribution targets — the fair neural-net baseline.

kd_04b's grid deliberately has no neural net: a PSSM is a per-position distribution, so sklearn's
plain-MSE MLPRegressor is the wrong model (it scored below the mean baseline — a strawman). The right
network outputs per-position logits and trains with cross-entropy. This runs that proper model
(`kmodel.dnn.PSSM_model`, the fastai MLP) under the SAME evaluation as kd_04b — repeated subfamily-grouped
CV over the full labelled set (REPEAT_SEEDS partitions) — and writes per-kinase scores in kd_04b's schema
so kd_04d can plot the sklearn grid and the DNN together.

Two **distribution** targets (softmax/CE only applies to distributions — the signed MLP-attribution
target has no DNN row here):
  pspa   scaled PSPA (kd_04b's PSPA target). kNN etc. already live in kd_04b's grid, so here the DNN
         adds only the MLP; scored on kd_04b's reduced flank columns so the numbers share one basis.
  cddm   CDDM substrate frequency (a per-position distribution), restricted to the ±5 flank. kd_04b's
         grid supplies kNN etc. and this adds only the MLP.

Out-of-fold predictions are produced for each of the REPEAT_SEEDS partitions (kd_04b's `repeat` column).
The DNN needs the full rectangular 23-residue x N-position PSSM as its target, so it trains on the full
grid but is scored over the ±5 flank (position-0 acceptor excluded, as everywhere in kd). Runs on this
Mac via MPS.

Inputs   out/kd_feat_{t5,onehot}.parquet (kd_02), kdata: pspa_scale, cddm, kinase_info
Outputs  out/kd_dnn_scores.parquet (per-kinase: target/feature/model/repeat/kd_ID/group/spearman/ap/pearson)

Run:  python nbs/kd_04c_dnn.py   (full; or --features onehot_pca,esm to recompute+merge only those)
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kd_util
import numpy as np
import pandas as pd
from paths import OUT

import kdata
from kmodel.dnn import PSSM_model, train_dl_cv
from kmodel.ml import get_splits

FEATURES = ['onehot', 'onehot_pca', 'esm', 't5']    # all four (match kd_04b) so the plots have no blanks
SPLIT_LEVEL, NFOLD = 'subfamily', 5
REPEAT_SEEDS = [0, 1, 2]                             # match kd_04b: distinct CV partitions, report per repeat
N_EPOCH, LR = 20, 3e-3
FLANK = list(range(-5, 6))                                   # ±5 including 0 (rectangular target for the DNN)


def _map_id():
    info = kdata.load('kinase_info')
    return info[info.pseudo == '0'].set_index('kinase')['kd_ID']


def flank_only(d):
    "Restrict a '{pos}{res}' PSSM to the ±5 positions (rectangular: same residue set at every position)."
    keep = [c for c in d.columns if (m := re.match(r'(-?\d+)', str(c))) and int(m.group(1)) in FLANK]
    return d[keep]


def load_target(name):
    "Return (target DataFrame keyed by kd_ID, scoring columns, model list) for a distribution target."
    id_map = _map_id()
    if name == 'pspa':
        d = kdata.load('pspa_scale')
        d = d[~d.index.str.contains('_')]                    # drop _TYR dual-specificity duplicates
        d.index = d.index.map(id_map)
        d = d[d.index.notna()]
        score_cols = kd_util.load_train('pspa', 'onehot')[2]  # kd_04b's reduced flank columns (one basis)
        return d, score_cols, ['MLP']                        # kNN for pspa comes from kd_04b's grid
    if name == 'cddm':
        d = flank_only(kdata.load('cddm'))                   # CDDM frequency, ±5 flank
        d.index = d.index.map(id_map)
        d = d[d.index.notna()]
        return d, list(d.columns), ['MLP']                   # kNN etc. come from kd_04b's CDDM grid
    raise ValueError(name)


def groups_of(ids):
    return kd_util.kinase_taxonomy(pd.Index(ids)).set_index('kinase')['group'].reindex(ids).to_numpy()


def score(target, feature, model, repeat, Yfull, Pfull, ridx, score_cols, ids):
    "Per-kinase flank Spearman/AP/Pearson on the scoring columns, tagged with kd_ID + group + repeat."
    sp, ap, pear = kd_util.pssm_scores(Yfull[:, ridx], Pfull[:, ridx], score_cols)
    return pd.DataFrame({'target': target, 'feature': feature, 'model': model, 'repeat': repeat,
                         'kd_ID': ids, 'group': groups_of(ids),
                         'spearman': sp, 'ap': ap, 'pearson': pear})


def main():
    import argparse
    p = argparse.ArgumentParser(description='kd_04c DNN (full run, or incremental subset merged in)')
    p.add_argument('--features', help='comma-separated features to recompute + merge (default: all)')
    p.add_argument('--targets', help='comma-separated targets to recompute + merge (default: pspa,cddm)')
    a = p.parse_args()
    targets = a.targets.split(',') if a.targets else ['pspa', 'cddm']
    features = a.features.split(',') if a.features else FEATURES
    incremental = bool(a.features or a.targets)

    rows = []
    for tname in targets:
        target, score_cols, models = load_target(tname)
        tcol = list(target.columns)
        ridx = [tcol.index(c) for c in score_cols]
        for feature in features:
            feat = pd.read_parquet(OUT / f'kd_feat_{feature}.parquet')
            df = target.join(feat, how='inner').reset_index()      # kd_ID back to a column (col 0)
            fcol = list(feat.columns)
            ids = df.iloc[:, 0].to_numpy()
            tax = kd_util.kinase_taxonomy(pd.Index(ids))           # row-aligned to df
            Y = df[tcol].to_numpy(float)

            for m in models:
                mk = lambda m=m: PSSM_model(len(fcol), len(tcol), model=m)
                for r, seed in enumerate(REPEAT_SEEDS):            # one OOF pass per CV partition
                    splits = list(get_splits(tax, stratified='group', group=SPLIT_LEVEL, nfold=NFOLD, seed=seed))
                    oof = train_dl_cv(df, fcol, tcol, splits, mk, n_epoch=N_EPOCH, lr=LR)
                    P = oof.loc[np.arange(len(df)), tcol].to_numpy(float)
                    rows.append(score(tname, feature, m, r, Y, P, ridx, score_cols, ids))
                print(f'  {tname:5} {feature:7} {m:5} done ({len(REPEAT_SEEDS)} repeats)', flush=True)

    new = pd.concat(rows, ignore_index=True)
    path = OUT / 'kd_dnn_scores.parquet'
    if incremental and path.exists():                            # replace exactly the recomputed cells
        old = pd.read_parquet(path)
        mask = old.target.isin(targets) & old.feature.isin(features)
        out = pd.concat([old[~mask], new], ignore_index=True)
        print(f'\nmerged {len(new)} recomputed rows into {path.name} ({len(old)} -> {len(out)})')
    else:
        out = new
    out.to_parquet(path)
    print('wrote', path, out.shape)
    print(out.groupby(['target', 'feature', 'model'])['spearman'].median().round(3).to_string())


if __name__ == '__main__':
    main()
