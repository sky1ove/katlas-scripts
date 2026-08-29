"""scoring_04a · PSPA scoring (generative), phospho ± percentile.

Ranks the fixed 295-kinase pool for every test site using the PSPA matrices, in two variants:
raw multiply score, and the same score converted to a percentile against the human-background
reference from scoring_01. Evaluated over the headline RepeatedStratifiedGroupKFold (K=5 × R=3):
every site is scored once per repeat, and the pairs stamp `seed = repeat`.

Generative means the PSSM is built (here: measured) independently of the ranking task — no
training on the split — so it is the natural baseline for the discriminative models in
scoring_04c. (PSPA is a fixed matrix, so nothing is refit per fold.)

Only the raw per-pair table is persisted; every metric is derived from it downstream.

Inputs   out/scoring_split.parquet, out/scoring_pool.parquet, out/pct_ref/PSPA_phospho.parquet,
         kdata.load('pspa')
Outputs  out/scoring_pairs/pspa_pairs.parquet

Run:  python nbs/scoring_04a_pspa.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import scoring_util as su

import kdata
from katlas.scoring import get_pct_df, multiply, predict_kinase_df


def main():
    split = su.load_split()
    pspa = kdata.load('pspa')
    site_kin = su.site_kinase_map(split)
    pools, _ = su.load_pool()
    common = pools['ST'] + pools['Tyr']
    pct = pd.read_parquet(su.PCT / 'PSPA_phospho.parquet')   # seed-independent

    res = []
    for rep, fold, _train, test in su.iter_folds(split):
        print(f'repeat {rep} fold {fold}')
        test = test[test.kinase_protein.isin(common)]
        st, tyr = su.branch_split(test)
        for branch, df_b in [('ST', st), ('Tyr', tyr)]:
            pool = pools[branch]
            r = predict_kinase_df(df_b, seq_col='site_seq',
                                  ref=pspa[pspa.index.isin(pool)], func=multiply
                                  ).reindex(columns=pool)
            for name, scores in [('PSPA: phospho', r),
                                 ('PSPA: phospho + pct', get_pct_df(r, pct).reindex(columns=pool))]:
                res.append(su.score_pairs(df_b, scores.to_numpy(np.float32), pool, site_kin,
                                          name, rep, branch))

    pairs = pd.concat(res, ignore_index=True)
    out = su.RES / 'pspa_pairs.parquet'
    pairs.to_parquet(out)
    print('saved', out, pairs.shape)
    print(su.summarize(pairs, ['branch', 'method']).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
