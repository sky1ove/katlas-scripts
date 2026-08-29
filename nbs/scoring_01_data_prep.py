"""scoring_01 · Data preparation for the kinase-substrate prediction benchmark.

Foundation for the whole scoring module: produces, once, the artifacts every later script loads.

1. **Stratified group split** (20% test × 3 seeds). Whole uppercase backbones (`site_seq_upper`)
   go entirely to train OR test, so no sequence is shared between them (leak-free), while each
   kinase keeps ~20% of its rows in test (`StratifiedGroupKFold`: group = backbone, stratify =
   kinase → per-kinase test fraction 0.200 ± 0.003, 0 backbone overlap). This supersedes the old
   per-kinase split, which shared ~91% of test sequences as "subfamily signal": the site-split
   ablation (archived `raw_scripts/scoring_12_site_split.py`) showed that leak is small on the
   num_kin ≤ 10 main subset, so the leak-free split is the main result.
   Saved as boolean `test_<seed>` columns (legacy, for window/num_kin) plus integer `fold_<repeat>`
   columns — the HEADLINE RepeatedStratifiedGroupKFold (K=5 × R=3): within a repeat the 5 folds tile
   the data (every site tested once, non-overlapping), the 3 repeats are independent re-shuffles, so
   the benchmark reports a reproducibility SD across full-CV repeats rather than a CI over three
   overlapping, non-independent draws.

2. **Fixed candidate pool** (295 = 219 ST + 76 Tyr), defined once on the full data and constant
   across seeds, so PSPA / CDDM motif / CDDM-seq all rank the identical set. Kinases with
   ≥ MIN_SITES sites after the promiscuity filter, non-pseudo, and present in PSPA.

3. **Baseline CDDM PSSMs** per seed, built on that seed's train split and reindexed to the pool.

4. **Percentile references** (human-background scores) for CDDM per seed and PSPA once — for the
   `+pct` variants. Phospho only; the uppercase refs/PSSMs are built in scoring_05a.

Inputs   kdata: ks_dataset(thr=None), pspa, kinase_info, human_site
Outputs  out/scoring_split.parquet, out/scoring_pool.parquet, out/scoring_cddm_seed{0,1,2}.parquet,
         out/pct_ref/PSPA_phospho.parquet, CDDM_phospho_seed{0,1,2}.parquet

Run:  python nbs/scoring_01_data_prep.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import scoring_util as su

import kdata
from katlas.scoring import multiply, multiply_23

FRAC = 0.2
FORCE_PCT = True    # refs depend on the split/CDDM; stale refs must never be silently reused


def load_sites():
    "All sites with num_kin, restricted to kinases with at least MIN_SITES of them."
    ks = kdata.ks_dataset(thr=None)
    ks['kinase_id'] = ks['kinase_uniprot'] + '_' + ks['kinase_protein'].str.split().str[0]
    vc = ks['kinase_id'].value_counts()
    ks = ks[ks['kinase_id'].isin(vc[vc >= su.MIN_SITES].index)].copy()
    ks['site_seq_upper'] = ks.site_seq.str.upper()
    print('rows:', len(ks), '| kinases:', ks.kinase_id.nunique())
    return ks


def add_splits(ks):
    """Two split schemes on one table:
    - LEGACY `test_<seed>` (overlapping 20% draws) — for the window (02*) and num_kin (03a) analyses.
    - HEADLINE `fold_<repeat>` (RepeatedStratifiedGroupKFold, K×R non-overlapping-within-repeat) — for
      the main benchmark (04*), so its CI is a reproducibility SD across full-CV repeats, not a
      parametric interval over three non-independent draws.
    Both are leak-free (whole backbones held out); asserts the backbone overlap is zero."""
    for seed in su.SEEDS:
        ks[f'test_{seed}'] = su.strat_group_split(ks, frac=FRAC, seed=seed)
        train_bb = set(ks.loc[~ks[f'test_{seed}'], 'site_seq_upper'])
        test_bb = set(ks.loc[ks[f'test_{seed}'], 'site_seq_upper'])
        overlap = len(train_bb & test_bb)
        print(f'seed {seed}: test={ks[f"test_{seed}"].sum()} '
              f'({100 * ks[f"test_{seed}"].mean():.0f}%) | backbone overlap={overlap}')
        assert overlap == 0, f'seed {seed} is not leak-free: {overlap} shared backbones'

    folds = su.repeated_group_folds(ks)                            # headline RepeatedSGKFold assignment
    for r in range(su.N_REPEATS):
        ks[f'fold_{r}'] = folds[f'fold_{r}'].to_numpy()            # positional (folds is 0..n-1 indexed)
        for k in range(su.N_FOLDS):                                # per-fold leak-free check
            te = ks[f'fold_{r}'] == k
            assert not (set(ks.loc[te, 'site_seq_upper']) & set(ks.loc[~te, 'site_seq_upper'])), \
                f'repeat {r} fold {k} shares backbones'
        print(f'repeat {r}: {su.N_FOLDS} folds, sizes={ks.groupby(f"fold_{r}").size().tolist()}')
    return ks


def save_split(ks):
    cols = (['kinase_uniprot', 'kinase_protein', 'kinase_group', 'kinase_id', 'source',
             'site_seq', 'site_seq_upper', 'sub_site', 'num_kin']
            + [f'test_{s}' for s in su.SEEDS] + [f'fold_{r}' for r in range(su.N_REPEATS)])
    split = ks[cols].copy()
    split.to_parquet(su.split_path())
    print('saved', su.split_path(), split.shape)
    return split


def build_pool(ks):
    "The fixed candidate pool: enough sites, non-pseudo, and measured by PSPA."
    pspa = kdata.load('pspa')
    info = kdata.load('kinase_info')
    info = info[info.pseudo == '0']
    info_ids = set(info.uniprot + '_' + info.kinase)

    vc = ks.loc[ks.num_kin <= su.BASE_NUMKIN, 'kinase_id'].value_counts()
    pool_ids = [k for k in vc[vc >= su.MIN_SITES].index if k in info_ids]
    names = pd.Series(pool_ids).str.split('_').str[1].drop_duplicates()

    pool = pd.DataFrame({'kinase': [k for k in names if k in set(pspa.index)]})
    pool['kinase_group'] = pool.kinase.map(info.set_index('kinase')['group'])
    pool['branch'] = np.where(pool.kinase_group == 'TK', 'Tyr', 'ST')
    pool.to_parquet(su.OUT / 'scoring_pool.parquet')
    print(f'fixed pool: {len(pool)} | ST {(pool.branch == "ST").sum()} '
          f'| Tyr {(pool.branch == "Tyr").sum()} -> {su.OUT / "scoring_pool.parquet"}')
    return pool.kinase.tolist()


def build_seed_artifacts(ks, pool):
    "Per-seed CDDM PSSM on the train split, plus its percentile reference."
    for seed in su.SEEDS:
        train = ks[~ks[f'test_{seed}']]
        cddm = su.build_pssm(train[train.num_kin <= su.BASE_NUMKIN], pool)
        uncovered = int(cddm.isna().all(axis=1).sum())
        assert cddm.shape[0] == len(pool), f'cddm not aligned to pool: {cddm.shape[0]} vs {len(pool)}'
        cddm.to_parquet(su.cddm_path(seed))
        print(f'seed {seed}: cddm {cddm.shape} | uncovered pool kinases: {uncovered}')
        su.cached_ref(f'CDDM_phospho_seed{seed}', cddm, 'site_seq', multiply_23, force=FORCE_PCT)


def main():
    su.OUT.mkdir(exist_ok=True)
    su.PCT.mkdir(parents=True, exist_ok=True)

    ks = add_splits(load_sites())
    save_split(ks)
    pool = build_pool(ks)

    # PSPA percentile reference is seed-independent, so it is computed once
    su.cached_ref('PSPA_phospho', kdata.load('pspa'), 'site_seq', multiply,
                  force=not (su.PCT / 'PSPA_phospho.parquet').exists())

    build_seed_artifacts(ks, pool)
    print('\nscoring_01 done.')


if __name__ == '__main__':
    main()
