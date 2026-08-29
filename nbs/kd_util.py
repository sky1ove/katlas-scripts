"""Shared helpers for the kd_04b+ scripts: data loading, taxonomy, and the retrieval model.

The kd model is a k-nearest-neighbour retrieval over the domain embeddings — for a query kinase,
its substrate PSSM is the inverse-distance-weighted mean of its nearest labelled kinases' PSSMs.
kd_04b chose it: on this small set (~370 labelled kinases) a 5-NN beat every parametric model and a
million-parameter CNN, which simply overfit. Retrieval also needs no training and no per-target
loss — the same code scores the PSPA distribution and the signed MLP-attribution PSSM — and its
nearest-neighbour distance is a natural confidence, used to decide which unknown domains to predict.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import OUT
from stats_util import nan_average_precision, nan_pearson, nan_spearman   # NaN-safe metric primitives

import kdata

#: feature name -> the prefix its columns carry in the joined training table
FEAT_PREFIX = {'t5': 'T5_', 'esm': 'ESM_', 'onehot': None, 'onehot_pca': None}

RETRIEVAL_K = 5     # neighbours for the k-NN retrieval model — a fixed default (performance is
                    # insensitive to k over {3,5,10}, <=0.01 flank Spearman); kd_04b's grid is fixed to it


def knn_predict(X_ref, Y_ref, X_query, k=RETRIEVAL_K):
    """Inverse-distance-weighted k-NN retrieval on raw features.

    Each query PSSM is the weighted mean of its `k` nearest reference kinases' PSSMs. Returns
    (prediction, nearest-neighbour distance); the distance is the per-query confidence — a small
    value means a close labelled relative exists, so the prediction can be trusted.
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k, len(X_ref))).fit(X_ref)
    dist, idx = nn.kneighbors(X_query)
    w = 1.0 / (dist + 1e-6)
    w /= w.sum(1, keepdims=True)
    return np.einsum('nk,nkd->nd', w, Y_ref[idx]), dist[:, 0]


def retrieval_oof(X, Y, splits, k=RETRIEVAL_K):
    "Out-of-fold k-NN predictions and each held-out kinase's nearest-in-training distance."
    P = np.zeros_like(Y)
    nnd = np.zeros(len(Y))
    for tr, va in splits:
        P[va], nnd[va] = knn_predict(X[tr], Y[tr], X[va], k)
    return P, nnd


def ridge_oof(X, Y, splits, alpha=1000.0):
    """Out-of-fold ridge predictions — the parametric baseline the retrieval model is checked against.

    Features are standardized per fold: ridge penalizes coefficients on the raw scale, so without it
    the (unstandardized) embedding dimensions are shrunk unevenly and the fit collapses to the mean.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    P = np.zeros_like(Y)
    for tr, va in splits:
        sc = StandardScaler().fit(X[tr])
        P[va] = Ridge(alpha=alpha).fit(sc.transform(X[tr]), Y[tr]).predict(sc.transform(X[va]))
    return P


# ---- deployment: the shipped predictor (configurable model), fit on the WHOLE labelled set ----------

def tuned_config(model_name, target, feature='onehot'):
    "Hyperparameters kd_04b tuned for (model, target, feature); {} if that cell was not in the grid."
    import ast
    path = OUT / 'kd_model_selection.parquet'
    if not path.exists():
        return {}
    sel = pd.read_parquet(path)
    row = sel[(sel.target == target) & (sel.feature == feature) & (sel.model == model_name)]
    if len(row) and isinstance(row.params.iloc[0], str) and row.params.iloc[0].strip():
        return ast.literal_eval(row.params.iloc[0])
    return {}


def build_estimator(model_name, target, feature='onehot'):
    """The deployment estimator for `model_name`, with kd_04b's tuned config (a fair regularised default
    when the cell was not tuned, e.g. a target outside the grid). Add a branch here to ship a new model."""
    cfg = tuned_config(model_name, target, feature)
    if model_name in ('RandomForest', 'ExtraTrees'):
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        cls = RandomForestRegressor if model_name == 'RandomForest' else ExtraTreesRegressor
        return cls(n_estimators=300, n_jobs=-1, random_state=123,
                   **{'max_features': 0.33, 'min_samples_leaf': 1, **cfg})
    if model_name == 'Ridge':
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), Ridge(**{'alpha': 1000, **cfg}))
    raise ValueError(f"no deployment estimator for '{model_name}' (kNN is handled separately)")


def deploy_predict(model_name, X_ref, Y_ref, X_query, target, feature='onehot', k=RETRIEVAL_K):
    """Fit `model_name` on the FULL labelled reference set and predict the queries; always return
    (prediction, nn_dist). nn_dist — the query's distance to the nearest labelled kinase — is the
    per-prediction confidence and is kept whatever the model is. 'kNN' uses inverse-distance retrieval;
    any other name is built by `build_estimator` and refit here on all of (X_ref, Y_ref)."""
    from sklearn.neighbors import NearestNeighbors
    nn_dist = NearestNeighbors(n_neighbors=1).fit(X_ref).kneighbors(X_query)[0][:, 0]
    if model_name == 'kNN':
        pred = knn_predict(X_ref, Y_ref, X_query, k)[0]
    else:
        est = build_estimator(model_name, target, feature).fit(X_ref, Y_ref)
        pred = np.asarray(est.predict(X_query)).reshape(len(X_query), -1)
    return pred, nn_dist


def average_precision(ref, pred, k=5):
    "AP@k (NaN-safe) — thin alias for stats_util.nan_average_precision, kept for the kd_* import path."
    return nan_average_precision(ref, pred, k)


def flank_cells(target_col):
    "Indices of the flanking cells (all positions except 0) and the per-flank-position residue groups."
    pos = np.array([int(re.match(r'(-?\d+)', str(c)).group(1)) for c in target_col])
    flank = np.where(pos != 0)[0]
    groups = [np.where(pos == p)[0] for p in np.unique(pos[pos != 0])]
    return flank, groups


def pssm_scores(Y, P, target_col):
    """Per-kinase agreement between true (Y) and predicted (P) PSSMs, over the FLANK only.

    Position 0 — the S/T/Y acceptor — is excluded from every metric: it is trivially set by the
    kinase group and its large cells would inflate the magnitude scores. Three per-kinase arrays:
      spearman  mean per-position Spearman (within-position preference rank)
      ap        AP@5 recovering the true flank's 5 strongest cells from the prediction's ranking
      pearson   overall Pearson over the flattened flank (whole-flank magnitude / linear agreement)
    """
    flank, groups = flank_cells(target_col)
    sp = np.full(len(Y), np.nan)
    ap = np.full(len(Y), np.nan)
    pear = np.full(len(Y), np.nan)
    for i in range(len(Y)):
        sp[i] = np.nanmean([nan_spearman(Y[i][g], P[i][g]) for g in groups if len(g) >= 3])
        ap[i] = nan_average_precision(Y[i][flank], P[i][flank], k=5)
        pear[i] = nan_pearson(Y[i][flank], P[i][flank])
    return sp, ap, pear


# ---------- confidence intervals ----------
# The kinase is the biological replicate for every kd metric (not the CV fold or the PSSM cell), so a CI
# is a bootstrap over the per-kinase score vector. The generic helper lives in stats_util (shared across
# modules); re-exported here so `kd_util.boot_ci` / `kd_util.fmt_ci` keep working for kd_04d / kd_09 / kd_11a.
from stats_util import boot_ci, fdr_bh, fmt_ci, mwu_report, wilcoxon_report  # noqa: E402,F401


def ranked_box(ax, per_item, xlabel, order=None, winner=None):
    """Horizontal box + strip of per-kinase scores by method/feature, ranked by median (best on top).

    `per_item` maps a name -> array of per-kinase scores. The selected entry (`winner`) is drawn in
    the accent colour with a bold label; the median is annotated at the right. Returns the plotted
    order so a companion panel can share it.
    """
    import seaborn as sns
    if order is None:
        order = sorted(per_item, key=lambda k: np.nanmedian(per_item[k]), reverse=True)
    long = pd.DataFrame([(m, v) for m in order for v in per_item[m]], columns=['name', 'v']).dropna()
    pal = {m: ('#c0392b' if m == winner else '#5b7fa6') for m in order}
    sns.boxplot(data=long, y='name', x='v', order=order, hue='name', palette=pal, legend=False,
                ax=ax, fliersize=0, width=0.6, linewidth=0.8)
    sns.stripplot(data=long, y='name', x='v', order=order, ax=ax, color='#2c3e50', size=1.4, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('')
    ax.axvline(0, color='0.8', lw=0.6, zorder=0)
    for i, m in enumerate(order):
        ax.text(1.01, i, f'{np.nanmedian(per_item[m]):.2f}', transform=ax.get_yaxis_transform(),
                va='center', ha='left', fontsize=8, color='#c0392b' if m == winner else '0.35')
    for lab in ax.get_yticklabels():
        if lab.get_text() == winner:
            lab.set_fontweight('bold')
            lab.set_color('#c0392b')
    return order


def load_train(target, feature):
    """One `kd_train_<target>_<feature>` table, split into (df, feature cols, target cols).

    The join in kd_03 puts the target PSSM columns first and the feature columns after, so the
    split is positional: everything from the feature table's width onwards is a feature.
    """
    path = OUT / f'kd_train_{target}_{feature}.parquet'
    if not path.exists():
        sys.exit(f'{path} not found - run kd_03_prepare_data.py first')
    df = pd.read_parquet(path)

    feat = pd.read_parquet(OUT / f'kd_feat_{feature}.parquet')
    feat_col = [c for c in df.columns if c in set(feat.columns)]
    target_col = [c for c in df.columns if c not in set(feat_col)]

    print(f'  {target}/{feature}: {df.shape} -> {len(feat_col)} features, {len(target_col)} targets')
    return df.reset_index(), feat_col, target_col


def kinase_taxonomy(kd_ids):
    "Per kd_ID subfamily / family / group, for building cross-validation splits."
    info = kdata.load('kinase_info')
    info = info[(info.pseudo == '0') & info.kd_ID.notna()]
    out = pd.DataFrame({'kinase': list(kd_ids)})
    for level in ['subfamily', 'family', 'group']:
        m = info[['kd_ID', level]].drop_duplicates().set_index('kd_ID')[level]
        out[level] = out.kinase.map(m)
    return out


def subfamily_palette():
    "subfamily -> the colour of its parent group, for the per-subfamily bar charts."
    from katlas.utils import group_color
    info = kdata.load('kinase_info')
    gc = pd.DataFrame(group_color).T.reset_index(names='group')
    pal = info[['group', 'subfamily']].merge(gc).drop(columns='group').set_index('subfamily')
    return pal.apply(tuple, axis=1).to_dict()
