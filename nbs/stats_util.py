"""Shared statistics helpers for the analysis scripts – currently the bootstrap confidence interval.

One home for the CI so every module reports it the same way. The rule across the whole pipeline: the
**biological replicate is the kinase**, never the CV fold (folds of one partition are not independent
experiments) and never the individual site / PSSM cell (pseudoreplicated – one kinase owns many). So a
CI is a bootstrap that resamples *kinases*. Scoring's micro leaderboard needs a cluster bootstrap over
kinases that also handles the per-site MAP – that lives next to its pair schema in `scoring_util`
(`boot_micro_ci`); everything else is a 1-D bootstrap of a per-kinase vector, which is `boot_ci` here.
"""
import numpy as np
import pandas as pd

DEFAULT_B = 2000        # bootstrap resamples – enough for a stable 2.5/97.5 percentile


# ---- NaN-safe agreement metrics (single source of truth) --------------------------------------------
# Every PSSM/vector comparison in the pipeline goes through these so NaN handling lives in ONE place.
# The rule: mask to cells finite in BOTH inputs *first*, then compute. Empirical PSSMs (PhosphoSitePlus
# sites), pspa_enrich (zeros→NaN) and surface-display matrices (masked low-n cells) carry NaN cells; a
# raw np.corrcoef / spearmanr / np.argsort over them silently voids the whole pair (corr → NaN) or lets
# NaN sort into the "top-k" set (argsort). Route comparisons here instead of re-deriving the mask ad hoc.

def _finite_pair(a, b):
    "The two vectors restricted to positions where BOTH are finite (drops NaN/inf pairwise)."
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def nan_pearson(a, b):
    "Pearson r over the pairwise-finite cells; NaN if <2 finite pairs or either side is constant."
    a, b = _finite_pair(a, b)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def nan_spearman(a, b):
    "Spearman rho over the pairwise-finite cells; NaN if <3 finite pairs."
    from scipy.stats import spearmanr
    a, b = _finite_pair(a, b)
    if len(a) < 3:
        return np.nan
    return float(spearmanr(a, b).correlation)


def nan_average_precision(ref, pred, k=5):
    """AP@k: recover `ref`'s k strongest cells from `pred`'s ranking (directional). Cells that are NaN in
    either `ref` or `pred` are dropped FIRST — otherwise NaN sorts to the top of the ranking and both the
    relevant set and the ranking get corrupted. Chance ≈ k / n_cells. NaN if no finite cells."""
    ref, pred = _finite_pair(ref, pred)
    if len(ref) == 0:
        return np.nan
    relevant = set(np.argsort(ref)[-k:])          # ref's k strongest cells (all of them if fewer than k)
    hits = ap = 0.0
    for rank, cell in enumerate(np.argsort(pred)[::-1], 1):
        if cell in relevant:
            hits += 1
            ap += hits / rank
    return ap / k


def boot_ci(vals, stat=None, B=DEFAULT_B, seed=0):
    """Percentile 95% CI of `stat` (default the median) over a 1-D sample, by ordinary bootstrap.

    `vals` is one value per kinase (the biological replicate), so resampling it with replacement is a
    bootstrap over kinases. Returns (lo, hi); (nan, nan) if fewer than 3 finite values.
    """
    v = pd.Series(vals).dropna().to_numpy()
    if len(v) < 3:
        return np.nan, np.nan
    stat = stat or np.median
    rng = np.random.default_rng(seed)
    bs = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(B)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_ci_cluster(values, groups, stat=None, B=DEFAULT_B, seed=0):
    """95% CI of `stat` (default the median) over a flat sample, resampling CLUSTERS (`groups`, e.g.
    kinase) with replacement and pooling the drawn clusters' rows – a cluster bootstrap.

    Use this when the point estimate is `stat(all rows)` but the rows are not independent (several per
    kinase, e.g. one per CV repeat): the point stays `stat(all rows)` while the CI resamples at the
    biological unit, so the two are mutually consistent. Returns (lo, hi); (nan, nan) if < 3 clusters.
    """
    v, g = np.asarray(values, float), np.asarray(groups)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    uniq = pd.unique(g)
    if len(uniq) < 3:
        return np.nan, np.nan
    arrs = [v[g == k] for k in uniq]                      # rows per cluster
    stat = stat or np.median
    rng = np.random.default_rng(seed)
    n = len(uniq)
    bs = np.array([stat(np.concatenate([arrs[i] for i in rng.integers(0, n, n)])) for _ in range(B)])
    return float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def boot_ci_corr(x, y, method='spearman', B=DEFAULT_B, seed=0):
    """95% CI of the correlation between paired arrays `x`, `y`, resampling the PAIRS (kinases) with
    replacement. `method` is 'spearman' or 'pearson'. Returns (lo, hi); (nan, nan) if < 3 finite pairs."""
    from scipy.stats import pearsonr, spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return np.nan, np.nan
    corr = ((lambda a, b: spearmanr(a, b).correlation) if method == 'spearman'
            else (lambda a, b: pearsonr(a, b)[0]))
    rng = np.random.default_rng(seed)
    bs = np.array([corr(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(B))])
    return float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def fmt_ci(v, ci, dp=3, cip=2):
    "Point estimate with its (lo, hi) 95% CI, e.g. '0.646 [0.55–0.72]'; bare value if the CI is nan."
    if ci is None or any(pd.isna(x) for x in ci):
        return f'{v:.{dp}f}'
    return f'{v:.{dp}f} [{ci[0]:.{cip}f}–{ci[1]:.{cip}f}]'


# ---------- null-hypothesis test reporting (Nature reporting summary) ----------
# Each test reports, where applicable: the test statistic, n / degrees of freedom, a standardized effect
# size, a 95% CI on the effect, and the exact (default two-sided) P. Wilcoxon/MWU are rank tests (no df).
def wilcoxon_report(a, b=None, alternative='two-sided', ci=True, seed=0):
    """Paired Wilcoxon signed-rank report. `a`,`b` are paired samples (or pass the differences as `a`
    with `b=None`, as in a one-sample signed-rank test). Returns a dict with the signed-rank statistic
    `W`, `n` non-zero pairs, the raw effect `median_delta` with a bootstrap 95% CI, the matched-pairs
    rank-biserial effect size `r` (Kerby 2014; −1..1), and the exact `p` for `alternative`."""
    from scipy.stats import rankdata, wilcoxon
    d = np.asarray(a, float) if b is None else np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    nz = d[d != 0]
    out = {'n': int(len(nz)), 'alternative': alternative,
           'median_delta': float(np.median(d)) if len(d) else np.nan,
           'ci': (boot_ci(d, np.median, seed=seed) if ci else (np.nan, np.nan))}
    if len(nz) < 1:
        out.update(W=np.nan, r=np.nan, p=np.nan)
        return out
    try:
        res = wilcoxon(d, alternative=alternative)
        ranks = rankdata(np.abs(nz))
        w_pos, w_neg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
        total = w_pos + w_neg
        out.update(W=float(res.statistic), p=float(res.pvalue),
                   r=float((w_pos - w_neg) / total) if total else np.nan)   # rank-biserial
    except ValueError:
        out.update(W=np.nan, r=np.nan, p=np.nan)
    return out


def mwu_report(x, y, alternative='two-sided'):
    """Mann-Whitney U report for two independent samples. Returns the U statistic, group sizes, the
    common-language effect size `auc` = U/(nx·ny) (the AUC / probability x > y), and the exact `p` for
    `alternative` ('two-sided', or 'greater'/'less' for a pre-specified directional hypothesis)."""
    from scipy.stats import mannwhitneyu
    x = np.asarray(x, float); y = np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 1 or len(y) < 1:
        return {'U': np.nan, 'nx': len(x), 'ny': len(y), 'auc': np.nan, 'p': np.nan, 'alternative': alternative}
    res = mannwhitneyu(x, y, alternative=alternative)
    return {'U': float(res.statistic), 'nx': int(len(x)), 'ny': int(len(y)),
            'auc': float(res.statistic / (len(x) * len(y))), 'p': float(res.pvalue), 'alternative': alternative}


def fdr_bh(pvals):
    """Benjamini–Hochberg FDR-adjusted p-values, returned in the input order (nan entries pass through).
    Use to correct a family of tests (e.g. every cell of a model grid) instead of raw per-test p."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return q
    order = np.argsort(pv)
    adj = pv[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]        # enforce monotone non-decreasing in rank
    inv = np.empty(m)
    inv[order] = np.clip(adj, 0, 1)
    q[ok] = inv
    return q
