"""kd_04a_tune · Grid-search the hyperparameters for every (model, feature, target) cell.

Split out from the evaluation (`kd_04b_model_comparison`): this does the ONE-TIME hyperparameter search
and writes the chosen config per cell; `kd_04b` then evaluates those FIXED configs by repeated
subfamily-grouped CV. Re-run only when a model's grid changes — the evaluation reads the frozen result.

For each cell, a single out-of-fold pass per candidate config on ONE subfamily-grouped, group-stratified
partition (seed `TUNE_SEED`), scored by the tuning objective (mean per-position flank Spearman over the
S/T kinases only — TK excluded, see kd_04b); the best config is kept. The estimators, grids, objective and
constants are imported from `kd_04b_model_comparison`, so there is a single source of truth.

Note this is only the *tuning* step. `TUNE_SEED` re-shuffles the SAME labelled cohort into folds — it is
NOT a disjoint tuning set (the same kinases are later scored in kd_04b), so the comparative grid scores may
be modestly optimistic. Acceptable here because the grid is descriptive: the shipped kNN×onehot (k=5) is
prespecified (chosen for deployment), not selected from it.

Inputs   out/kd_train_{pspa,cddm,mlp_attr}_{onehot,onehot_pca,esm,t5}.parquet (kd_03)
Outputs  out/kd_tuned_params.parquet (target, feature, model, params)

Run:  python nbs/kd_04a_tune.py   (full; or --models/--features/--targets for a subset, merged in)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kd_util
import numpy as np
import pandas as pd
from paths import OUT
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, cross_val_predict

from kmodel.ml import get_splits
import kd_04b_model_comparison as k5

TUNE_SEED = 100                          # the single partition on which hyperparameters are chosen —
                                         # deliberately OUTSIDE kd_04b's REPEAT_SEEDS=[0,1,2] so no
                                         # evaluation fold is ever reused for tuning (the config is fixed
                                         # a priori on an independent partition; this is not nested CV)


def tune_cell(name, est, grid, df, feat_col, target_col, splits, gmap):
    "One out-of-fold pass per candidate config; keep the best by the S/T flank objective (k5.cv_objective)."
    X, Y = df[feat_col].to_numpy(float), df[target_col].to_numpy(float)
    grp = pd.Series(df.iloc[:, 0].to_numpy()).map(gmap).to_numpy()
    flank, groups = kd_util.flank_cells(target_col)
    cv_jobs = 1 if name in k5.HEAVY_INTERNAL else -1
    sel = ~np.isin(grp, k5.SELECT_EXCLUDE)                    # kinases the tuning objective sees (S/T)
    best = (-np.inf, {})
    for cfg in ParameterGrid(grid):                          # ParameterGrid({}) yields one empty config
        Poof = cross_val_predict(clone(est).set_params(**cfg), X, Y, cv=splits, n_jobs=cv_jobs)
        obj = k5.cv_objective(Y[sel], Poof[sel], flank, groups, mode=k5.TUNE_ON)
        if obj > best[0]:
            best = (obj, cfg)
    return best[1]


def main():
    p = argparse.ArgumentParser(description='kd_04a_tune grid search (full, or incremental subset merged in)')
    p.add_argument('--models', help='comma-separated models to recompute + merge (default: all)')
    p.add_argument('--features', help='comma-separated features (default: all)')
    p.add_argument('--targets', help='comma-separated targets (default: all)')
    a = p.parse_args()
    csv = lambda s: s.split(',') if s else None
    sel_models, sel_features, sel_targets = csv(a.models), csv(a.features), csv(a.targets)
    targets = sel_targets or k5.TARGETS
    features = sel_features or k5.FEATURES
    incremental = bool(sel_models or sel_features or sel_targets)

    rows = []
    for label in targets:
        print(f'\n== {label.upper()} ==')
        gmap = None
        for feature in features:
            df, feat_col, target_col = kd_util.load_train(label, feature)
            info = kd_util.kinase_taxonomy(df.iloc[:, 0])
            if gmap is None:
                gmap = info.set_index('kinase')['group']
            splits = list(get_splits(info, stratified='group', group=k5.SPLIT_LEVEL,
                                     nfold=k5.NFOLD, seed=TUNE_SEED))
            for name, (est, grid) in k5.models_and_grids().items():
                if sel_models and name not in sel_models:
                    continue
                params = tune_cell(name, est, grid, df, feat_col, target_col, splits, gmap)
                rows.append({'target': label, 'feature': feature, 'model': name, 'params': str(params)})
                print(f'  {feature:11} {name:13} {params}', flush=True)

    new = pd.DataFrame(rows)
    path = OUT / 'kd_tuned_params.parquet'
    key = ['target', 'feature', 'model']
    if incremental and path.exists():                        # replace exactly the recomputed cells
        old = pd.read_parquet(path)
        keep = old.merge(new[key], on=key, how='left', indicator=True).query('_merge == "left_only"')
        new = pd.concat([keep.drop(columns='_merge'), new], ignore_index=True)
        print(f'\nmerged {len(rows)} recomputed cells into {path.name}')
    new.to_parquet(path)
    print('\nwrote', path, new.shape)


if __name__ == '__main__':
    main()
