"""scoring_04b · CDDM motif scoring (generative) at the selected ±5 window.

The paper's "CDDM" row. Over the headline RepeatedStratifiedGroupKFold (K=5 × R=3), the CDDM PSSM
is rebuilt on each fold's own train split (leak-free) and restricted to positions −5…+5 — the
window scoring_02b's validation sweep selects, and the same one the discriminative models use in
scoring_04c, so generative and discriminative are compared at one window. Pairs stamp `seed = repeat`.

Two variants: the raw multiply score and its percentile against a human background scored with
the ±5 PSSM (recomputed per fold so the percentile is on the right scale and leak-free).

Inputs   out/scoring_split.parquet, out/scoring_pool.parquet
Outputs  out/scoring_pairs/cddm_pairs.parquet, out/pct_ref/CDDM_w5_phospho_rep*_fold*.parquet

Run:  python nbs/scoring_04b_cddm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import scoring_util as su

from katlas.scoring import get_pct_df, multiply_23, predict_kinase_df

WINDOW = 5          # scoring_02b's validation argmax; ±5-12 is a flat plateau, ±5 = shortest interpretable motif
FORCE_PCT = True    # the w5 refs depend on the split/CDDM - recompute so stale refs are never reused


def main():
    split = su.load_split()
    site_kin = su.site_kinase_map(split)
    pools, _ = su.load_pool()
    common = pools['ST'] + pools['Tyr']

    res = []
    for rep, fold, train, test in su.iter_folds(split):
        print(f'repeat {rep} fold {fold}')
        cddm = su.build_pssm(train[train.num_kin <= su.BASE_NUMKIN], common)   # leak-free per-fold CDDM
        cddm_w = su.win(cddm, WINDOW)
        ref_pct = su.cached_ref(f'CDDM_w5_phospho_rep{rep}_fold{fold}', cddm_w, 'site_seq',
                                multiply_23, force=FORCE_PCT)

        test = test[test.kinase_protein.isin(common)]
        st, tyr = su.branch_split(test)
        for branch, te in [('ST', st), ('Tyr', tyr)]:
            pool = pools[branch]
            r = predict_kinase_df(te, seq_col='site_seq',
                                  ref=cddm_w[cddm_w.index.isin(pool)], func=multiply_23
                                  ).reindex(columns=pool)
            for name, scores in [('CDDM (w5)', r),
                                 ('CDDM (w5) pct', get_pct_df(r, ref_pct).reindex(columns=pool))]:
                res.append(su.score_pairs(te, scores.to_numpy(np.float32), pool, site_kin,
                                          name, rep, branch))

    pairs = pd.concat(res, ignore_index=True)
    out = su.RES / 'cddm_pairs.parquet'
    pairs.to_parquet(out)
    print('saved', out, pairs.shape, '| methods:', sorted(pairs.method.unique()))
    print(su.summarize(pairs, ['branch', 'method']).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
