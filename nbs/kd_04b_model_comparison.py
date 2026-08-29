"""kd_04b · Model x feature grid, evaluated by repeated subfamily-grouped CV over the full labelled set.

Trains the grid and persists the per-kinase scores; kd_04d draws the comparison figures — this script
does no plotting (the "persist raw scores, derive views" invariant: a figure or slice is a filter over
the saved table, never a re-score). The grid covers the feature axis too, so there is no separate
feature-selection step: the shipped feature falls out of the same table.

**No held-out test set.** kNN was chosen for deployment reasons (interpretable retrieval, a free
confidence distance, best in the high-similarity regime kd_07 ships into) — not by topping this grid —
so the grid is *descriptive* (show kNN is competitive), not the mechanism that selects the model. With
nothing being selected on the reported number, a small held-out test would only add variance (~56
kinases is noisy), so instead every cell is scored by **out-of-fold cross-validation over the target's
whole labelled set**, and to damp the partition-dependence of one arbitrary fold split, the whole CV is
**repeated `REPEATS` times** with different partitions (`REPEAT_SEEDS`). Each split is subfamily-
grouped and group-stratified — the same anti-leakage scheme as everywhere in kd (whole subfamilies stay
on one side; groups stay proportionally represented).

**Tuning is separated out (kd_04a_tune).** The hyperparameter grid search runs once in `kd_04a_tune`,
which writes the chosen config per cell to `out/kd_tuned_params.parquet`; **this script only evaluates
those FIXED configs**. Tuning uses a separate partition (`TUNE_SEED=100`) but of the **same** labelled
cohort — **not a disjoint tuning set** (the same kinases are scored here), so the comparative grid scores
may be **modestly optimistic**; that is acceptable because the grid is descriptive and the shipped
kNN×onehot (`k=5`) is **prespecified**, not selected from it. For every (regressor, feature) cell: read its fixed config,
then produce out-of-fold predictions on **each** of the `REPEATS` partitions. Per-kinase scores are
saved for every (cell, repeat) with each kinase's `group`; kd_04d reads them — point estimate = mean
over kinases and repeats, error bar = std **across repeats**, and "is cell A really better than the
selected kNN×onehot" is a **paired** test across the matched per-kinase OOF scores.

**`SELECT_EXCLUDE` groups (default TK) are excluded from the tuning objective (in kd_04a_tune) and from
the best-cell summary here.** Tyrosine kinases are internally homogeneous — their group-mean motif
already scores as high as retrieval, so absolute Spearman on TK is inflated without testing per-kinase
specificity. Every kinase is still scored (the `group` column lets kd_04d slice); only the *decisions*
ignore TK. `models_and_grids` (with `cv_objective`) is shared with `kd_04a_tune` so there is one source
of truth for the estimators and grids.

Models (rows):
  mean                              training-mean PSSM (the floor; no hyperparameters)
  Linear / Ridge / Lasso / ElasticNet   standardized linear models — full sweep (alpha / l1_ratio)
  PLS                               latent-variable linear — full sweep (n_components)
  kNN                               inverse-distance retrieval — full sweep (k); kd_util.knn_predict
  DecisionTree                      tree — full sweep (max_depth)
  SVR                               RBF support-vector regression (C), per-position — minimal grid
Features (columns): onehot, onehot_pca, esm, t5 (kd_02a one-hot / kd_02b protein-LM embeddings).

**No neural net here.** A PSSM is a per-position distribution, so sklearn's plain-MSE MLPRegressor is
the wrong model (below the mean baseline) — a strawman, not evidence. The proper softmax/CE DNN (fastai
MLP/CNN, kd_04c) is a fair baseline: it ties k-NN on the distribution metric and trails it on flank
Spearman. kd_04d reads both this grid and kd_04c's DNN scores into the combined figures/tables.

Scored over the ±5 flank (position-0 acceptor excluded — group-determined, its large cells inflate the
magnitude metrics) by three per-kinase measures: mean per-position Spearman, AP@5 (recover the real
PSSM's five strongest flank cells from the prediction's ranking), and overall flank Pearson. All three
work for the PSPA and CDDM-frequency distributions and for the signed MLP-attribution target (all
three are scored by the same rank/magnitude measures over the flank).

The evaluation is run by `run_grid()`, over the fixed configs `kd_04a_tune` wrote.

Inputs   out/kd_tuned_params.parquet (kd_04a_tune — the fixed configs),
         out/kd_train_{pspa,cddm,mlp_attr}_{onehot,onehot_pca,esm,t5}.parquet (kd_03)
Outputs  out/kd_grid_scores.parquet (per-kinase, long, one row per cell x repeat x kinase — kd_04d reads
         this), out/kd_model_selection.parquet (median summary + the fixed params)

Run:  python nbs/kd_04a_tune.py  then  python nbs/kd_04b_model_comparison.py
      (--models Ridge,kNN etc. recompute+merge only those)
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kd_util
import numpy as np
import pandas as pd
from paths import OUT
from sklearn.base import BaseEstimator, RegressorMixin

from kmodel.ml import get_splits

FEATURES = ['onehot', 'onehot_pca', 'esm', 't5']
TARGETS = ['pspa', 'cddm', 'mlp_attr']  # two distributions (pspa scaled, cddm frequency) + the signed
                                        # count-robust MLP-attribution target; each scored on its own
                                        # full labelled set (no shared split couples them)
SPLIT_LEVEL, NFOLD = 'subfamily', 5     # subfamily-grouped, group-stratified K-fold
REPEAT_SEEDS = [0, 1, 2]                # REPEATS distinct CV partitions; report mean + across-repeat std
METRICS = ['spearman', 'ap', 'pearson']
#: hyperparameter-tuning objective — 'spearman' (primary specificity metric, recommended), 'pearson'
#: (smoother magnitude signal), or 'both' (their mean). Selection/reporting keep all three metrics.
TUNE_ON = 'spearman'
#: kinase groups excluded from the tuning objective + selection (NOT from the persisted scores — every
#: kinase is still trained on and scored, and its group is stored). Tyrosine kinases (TK) are internally
#: homogeneous: their group-mean motif already scores as high as retrieval, so absolute Spearman on TK
#: is inflated without testing per-kinase specificity. Selection targets S/T specificity. Set () to
#: select on all kinases.
SELECT_EXCLUDE = ('TK',)
SHIPPED = ('kNN', 'onehot')     # the (model, feature) kd_07 deploys — used for the run-log print
#: models that parallelise internally (n_jobs=-1) — their CV folds run serially to avoid nested
#: oversubscription; every other model parallelises across the 5 CV folds instead
HEAVY_INTERNAL = {'SVR'}


# module-level (not nested in models_and_grids) so joblib/loky can pickle them for n_jobs parallelism
class MeanRegressor(BaseEstimator, RegressorMixin):
    "Predict the per-column training mean — the floor baseline."
    def fit(self, X, y):
        self.mean_ = np.asarray(y, float).mean(0)
        return self
    def predict(self, X):
        return np.tile(self.mean_, (len(X), 1))


class KNNRetriever(BaseEstimator, RegressorMixin):
    "Inverse-distance k-NN retrieval (kd_util.knn_predict) wrapped so k is a tunable hyperparameter."
    def __init__(self, k=5):
        self.k = k
    def fit(self, X, y):
        self.Xr_, self.Yr_ = np.asarray(X, float), np.asarray(y, float)
        return self
    def predict(self, X):
        return kd_util.knn_predict(self.Xr_, self.Yr_, np.asarray(X, float), self.k)[0]


def models_and_grids():
    "Each model as an sklearn estimator + its compact hyperparameter grid (empty = nothing to tune)."
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    def scaled(model):
        return make_pipeline(StandardScaler(), model)

    return {
        'mean': (MeanRegressor(), {}),
        'Linear': (scaled(LinearRegression()), {}),
        'Ridge': (scaled(Ridge()), {'ridge__alpha': [1, 10, 100, 1000, 1e4]}),
        'Lasso': (scaled(Lasso(max_iter=5000)), {'lasso__alpha': [1e-3, 1e-2, 1e-1]}),
        'ElasticNet': (scaled(ElasticNet(max_iter=5000)),
                       {'elasticnet__alpha': [1e-3, 1e-2, 1e-1], 'elasticnet__l1_ratio': [0.2, 0.5, 0.8]}),
        'PLS': (PLSRegression(), {'n_components': [5, 10, 20, 40]}),
        'kNN': (KNNRetriever(), {'k': [5]}),   # fixed default: k is insensitive over {3,5,10} (<=0.01),
        #                                        and k=5 matches deployment (kd_util.RETRIEVAL_K)
        # SVR is a compute-heavy reference baseline (minimal grid). It carries heavy *internal*
        # parallelism (one SVR per output column) so it runs n_jobs=-1 with CV folds serial (see
        # HEAVY_INTERNAL); the light models parallelise across CV folds instead.
        # epsilon 0.05 on the [0,1] target scale (the 0.1 default underfits); one SVR per output column.
        'SVR': (scaled(MultiOutputRegressor(SVR(epsilon=0.05), n_jobs=-1)),
                {'multioutputregressor__estimator__C': [1, 10]}),
        'DecisionTree': (DecisionTreeRegressor(),
                         {'max_depth': [4, 8, 16, None], 'min_samples_leaf': [1, 3]}),
        # No neural net here: a PSSM is a per-position distribution, so sklearn's plain-MSE MLPRegressor
        # is the wrong model (it scored below the mean baseline). The proper softmax/CE DNN (fastai
        # MLP/CNN) is compared separately in kd_04c — that is the fair neural-net baseline.
    }


def cv_objective(Y, P, flank, groups, mode=None):
    """Hyperparameter-tuning objective, meaned over kinases. Selection/reporting still show all three
    metrics regardless — this only decides which config a model keeps.

    mode (TUNE_ON): 'spearman' — mean per-position flank Spearman (rank; the primary specificity
    metric, scale-invariant across targets — recommended); 'pearson' — overall flank Pearson (smoother
    magnitude signal, but not comparable across differently-scaled targets); 'both' — their mean.
    """
    from scipy.stats import spearmanr
    mode = mode or TUNE_ON

    def sp_i(i):
        return np.nanmean([spearmanr(Y[i][g], P[i][g]).correlation for g in groups if len(g) >= 3])

    def pe_i(i):
        return np.corrcoef(Y[i][flank], P[i][flank])[0, 1]

    if mode == 'spearman':
        vals = [sp_i(i) for i in range(len(Y))]
    elif mode == 'pearson':
        vals = [pe_i(i) for i in range(len(Y))]
    else:
        vals = [0.5 * (sp_i(i) + pe_i(i)) for i in range(len(Y))]
    return float(np.nanmean(vals))


def run_cell(label, feature, name, est, params, df, feat_col, target_col, splits_by_seed, gmap):
    """Score a FIXED hyperparameter config (from kd_04a_tune) by repeated subfamily-grouped CV.

    No tuning here — the config is fixed a priori (kd_04a_tune's grid search runs on TUNE_SEED=100). This
    is repeated subfamily-grouped CV with pre-selected hyperparameters, NOT nested CV (no inner-fold tuning
    inside each outer fold). TUNE_SEED=100 only re-shuffles the SAME cohort (not a disjoint tuning set), so
    the grid scores may be modestly optimistic — fine here because the grid is descriptive and the shipped
    kNN×onehot (k=5) is prespecified. Every kinase is scored on each of the
    REPEAT_SEEDS partitions; the point estimate and across-repeat std come out of the persisted per-kinase rows.
    """
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_predict
    X, Y = df[feat_col].to_numpy(float), df[target_col].to_numpy(float)
    ids = df.iloc[:, 0].to_numpy()
    grp = pd.Series(ids).map(gmap).to_numpy()
    cv_jobs = 1 if name in HEAVY_INTERNAL else -1            # heavy models parallelise inside the fit
    base = clone(est).set_params(**params)                   # the fixed config for this cell

    rows = []
    for r, splits in enumerate(splits_by_seed):
        P = cross_val_predict(clone(base), X, Y, cv=splits, n_jobs=cv_jobs)
        sp, ap, pear = kd_util.pssm_scores(Y, P, target_col)
        rows.append(pd.DataFrame({'target': label, 'feature': feature, 'model': name, 'repeat': r,
                                  'kd_ID': ids, 'group': grp,
                                  'spearman': sp, 'ap': ap, 'pearson': pear}))
    return pd.concat(rows, ignore_index=True), str(params)


def rebuild_summary(grid, param_rows, suffix):
    "Median summary + tuned params, derived from the (possibly merged) per-kinase grid."
    key = ['target', 'feature', 'model']
    new_p = pd.DataFrame(param_rows)
    old_path = OUT / f'kd_model_selection{suffix}.parquet'
    if old_path.exists():                                        # carry params for cells not recomputed
        oldp = pd.read_parquet(old_path)[key + ['params']].drop_duplicates()
        keep = oldp.merge(new_p[key], on=key, how='left', indicator=True).query('_merge == "left_only"')
        params = pd.concat([keep[key + ['params']], new_p], ignore_index=True)
    else:
        params = new_p
    long = grid.melt(id_vars=['target', 'feature', 'model', 'repeat', 'kd_ID'],
                     value_vars=METRICS, var_name='metric', value_name='value')
    summary = (long.groupby(['target', 'feature', 'model', 'metric'])['value'].median()
               .round(3).unstack('metric').reset_index())
    frac = (grid.groupby(key)['spearman']
            .apply(lambda s: round(float(np.mean(s > 0.3)), 2)).rename('cv_frac>0.3').reset_index())
    summary = summary.merge(frac, on=key, how='left').merge(params, on=key, how='left')
    summary.to_parquet(old_path)


def load_params():
    "kd_04a_tune's chosen config per (target, feature, model) — the FIXED hyperparameters we evaluate."
    path = OUT / 'kd_tuned_params.parquet'
    if not path.exists():
        sys.exit(f'{path} not found - run kd_04a_tune.py first (the grid search)')
    d = pd.read_parquet(path)
    return {(r.target, r.feature, r.model): ast.literal_eval(r.params) for r in d.itertuples()}


def run_grid(suffix='', sel_models=None, sel_features=None, sel_targets=None):
    """Evaluate every (model, feature) cell at its FIXED (kd_04a_tune) config by repeated subfamily-grouped
    CV, writing kd_grid_scores{suffix} + kd_model_selection{suffix}.

    Incremental: passing any of sel_models / sel_features / sel_targets recomputes only that subset and
    MERGES it into the existing outputs (replacing exactly those cells). With no selectors it is a full run.
    """
    targets, features = sel_targets or TARGETS, sel_features or FEATURES
    incremental = bool(sel_models or sel_features or sel_targets)
    param_lookup = load_params()

    all_scores, param_rows = [], []
    for label in targets:
        print(f'\n== {label.upper()} ==')
        gmap = None
        for feature in features:
            df, feat_col, target_col = kd_util.load_train(label, feature)
            info = kd_util.kinase_taxonomy(df.iloc[:, 0])
            if gmap is None:                                 # kd_ID -> group (same kinase set per feature)
                gmap = info.set_index('kinase')['group']
            splits_by_seed = [list(get_splits(info, stratified='group', group=SPLIT_LEVEL,
                                              nfold=NFOLD, seed=s)) for s in REPEAT_SEEDS]
            for name, (est, _grid) in models_and_grids().items():
                if sel_models and name not in sel_models:
                    continue
                params = param_lookup.get((label, feature, name), {})
                scores, params_str = run_cell(label, feature, name, est, params, df,
                                              feat_col, target_col, splits_by_seed, gmap)
                all_scores.append(scores)
                param_rows.append({'target': label, 'feature': feature, 'model': name, 'params': params_str})
                print(f'  {feature:11} {name:13} {params_str}', flush=True)

    new = pd.concat(all_scores, ignore_index=True)
    path = OUT / f'kd_grid_scores{suffix}.parquet'
    if incremental and path.exists():
        old = pd.read_parquet(path)
        mask = old.target.isin(targets) & old.feature.isin(features)      # replace exactly the recomputed
        if sel_models:
            mask &= old.model.isin(sel_models)
        grid = pd.concat([old[~mask], new], ignore_index=True)
        print(f'\nmerged {len(new)} recomputed rows into {path.name} ({len(old)} -> {len(grid)})')
    else:
        grid = new
    grid.to_parquet(path)
    print('wrote', path, grid.shape)

    rebuild_summary(grid, param_rows, suffix)

    on = 'all kinases' if not SELECT_EXCLUDE else 'non-' + '/'.join(SELECT_EXCLUDE) + ' kinases'
    for label in [t for t in TARGETS if t in grid.target.unique()]:
        print(f'\n== {label.upper()} — median CV Spearman on {on} (fixed configs) ==')
        sub = grid[(grid.target == label) & (~grid['group'].isin(SELECT_EXCLUDE))]
        piv = sub.groupby(['model', 'feature'])['spearman'].median()
        best = piv.idxmax()
        shipped = piv.get(SHIPPED, float('nan'))
        print(f'  best CV cell: {best[0]} x {best[1]} ({piv.max():.3f}); '
              f'shipped {SHIPPED[0]} x {SHIPPED[1]} ({shipped:.3f})')
    print('\nwrote', OUT / f'kd_model_selection{suffix}.parquet')
    return grid


def main():
    import argparse
    p = argparse.ArgumentParser(description='kd_04b model x feature grid (full run, or incremental subset)')
    p.add_argument('--models', help='comma-separated models to recompute + merge (default: all)')
    p.add_argument('--features', help='comma-separated features to recompute + merge (default: all)')
    p.add_argument('--targets', help='comma-separated targets to recompute + merge (default: all)')
    a = p.parse_args()
    csv = lambda s: s.split(',') if s else None
    run_grid(sel_models=csv(a.models), sel_features=csv(a.features), sel_targets=csv(a.targets))


if __name__ == '__main__':
    main()
