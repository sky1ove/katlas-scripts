"""kd_06b · Calibrate the proximity cutoff: how close must a domain be to a labelled kinase to
predict its motif well? (The output file is still named kd_confidence_threshold.* for back-compat; the
paper term is "proximity cutoff" — this is a distance cutoff + proximity tiers, not calibrated confidence.)

The deliverable is not "predict every uncharacterised domain" — it is "predict the motif of a domain
that is *very similar* to a known kinase, reliably." So we need a proximity cutoff: below it, retrieval
from the nearest labelled kinase is trustworthy; above it, we do not claim a prediction (a "within-cutoff"
prediction is one that passes it).

Calibration choices:
  * accuracy vs nn_dist is shown for **both** overall flank Pearson (magnitude — stable for very-specific
    kinases whose per-position ranking is degenerate) and per-position Spearman (within-position rank).
    The threshold is the **Kneedle elbow** (Satopää et al. 2011) of the `THRESHOLD_METRIC` curve — the
    normalised curve's point of maximum distance below the first->last chord, i.e. where accuracy stops
    being flat and starts dropping. It is a deterministic algorithm with no arbitrary accuracy bar, and
    its stability is quantified by a **bootstrap 95% CI** (resample kinases -> recompute the elbow, BOOT
    times); the accuracy curve annotates the **kinase count per bin**. Each target's own elbow is shown,
    but ONE conservative cutoff (THRESHOLD_TARGET's) is applied to all so a domain's predictability does
    not flip between methods on curve noise.
  * calibrate on **S/T kinases**, look at TK separately. TK are internally homogeneous — any TK's motif
    resembles any other's — so they stay "well predicted" at large nn_dist and would distort the S/T
    curve; they are drawn but excluded from the threshold, and for MLP-attr (its TK signal is unstable —
    the threshold curve shows TK poorly recovered) TK are not predicted at all.
  * the elbow is a single keep/drop line, but predictions inside it are not equally close to a labelled
    kinase, so each is also **graded by proximity**. The predictable region [0, cut] is split into three
    equal **distance** bands — high / intermediate / low **proximity**. These are PROXIMITY tiers (raw
    nn_dist), NOT empirically calibrated confidence tiers: flank-Pearson is not comparable across methods,
    so an absolute accuracy level would mislead, and a domain's proximity is only a monotone proxy for
    reliability, not a validated accuracy. 'low proximity' is the third just below the elbow. The bands are
    shared, but each method grades a domain by ITS OWN nn_dist (a different labelled set), so the same
    domain can land in different tiers per method. Rides into kd_07's predictions as `proximity_tier`.

Calibrated by **leave-one-out** k-NN (NOT the subfamily-grouped CV of kd_04b): deployment keeps a
domain's close labelled relatives in the reference, so calibration must too. For each labelled kinase we
get (nn_dist to its nearest *other* labelled kinase, LOO retrieval accuracy). nn_dist is the same one-hot
distance kd_07 attaches to each uncharacterised domain, so the cutoff transfers directly; kd_07's
predictions are annotated in place with `predictable` (and its `organism` / `is_human` flags let the
dark human kinome be read apart from the cross-species orthologs).

Inputs   out/kd_train_{pspa,mlp_attr,cddm}_onehot.parquet (kd_03), out/kd_feat_onehot.parquet (kd_02a),
         out/kd_pred_new_{pspa,mlp_attr,cddm}.parquet (kd_07)
Outputs  out/kd_confidence_threshold.parquet (per-target threshold + proximity-band boundaries + coverage),
         fig/kd_confidence_threshold.svg, and `predictable` / `proximity_tier` written into kd_pred_new_*

Run:  python nbs/kd_06b_confidence_threshold.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kd_util
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from paths import FIG, OUT

from kplot.utils import save_svg, set_sns

TARGETS = ['pspa', 'cddm', 'mlp_attr']
K = 5                              # retrieval neighbours (kd_07's k)
NBIN = 10                         # nn_dist quantile bins for the accuracy curve
BOOT = 1000                       # bootstrap resamples for the elbow / threshold confidence interval
THRESHOLD_METRIC = 'pearson'      # which curve's elbow sets the threshold ('pearson' or 'spearman')
THRESHOLD_TARGET = 'pspa'         # ONE threshold for all targets = this target's elbow (the most
                                  # conservative — PSPA's cutoff is the tightest, so we do not over-claim)
DROP_TK = {'mlp_attr'}            # targets where TK is not predicted — its MLP-attr signal is unstable
                                  # (the threshold curve shows TK scattered / not reliably recovered)
METRICS = [('overall flank Pearson', 2), ('per-position Spearman', 0)]   # (label, pssm_scores index)


def loo_retrieval(X, Y, k=K):
    "Leave-one-out k-NN: predict each kinase from the OTHER labelled kinases; return (pred, nn_dist)."
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)             # +1 because the nearest is self
    dist, idx = nn.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]                         # drop self
    w = 1.0 / (dist + 1e-6)
    w /= w.sum(1, keepdims=True)
    return np.einsum('nk,nkd->nd', w, Y[idx]), dist[:, 0]


def binned_curve(nn_dist, score):
    "Median `score` (+ kinase count n) in nn_dist quantile bins — the accuracy-vs-distance curve."
    b = pd.DataFrame({'d': nn_dist, 's': score})
    b['bin'] = pd.cut(b.d, np.unique(np.quantile(b.d, np.linspace(0, 1, NBIN + 1))), include_lowest=True)
    return (b.groupby('bin', observed=True)
            .agg(hi=('d', 'max'), s=('s', 'median'), n=('s', 'size')).reset_index())


def elbow(curve):
    """Kneedle elbow (Satopää et al. 2011): normalise the (nn_dist, accuracy) curve to [0,1] and return
    the nn_dist of the point of maximum perpendicular distance from the first->last chord — for a curve
    that stays high then drops, this knee is where the flat-high accuracy turns into the decline (the
    curve bulges away from the chord there). Deterministic — no accuracy threshold to choose. Guard: with
    <3 bins or a near-flat curve (range < 0.05) accuracy never really drops, so retrieval is reliable throughout."""
    x, y = curve.hi.to_numpy(float), curve.s.to_numpy(float)
    if len(x) < 3 or (y.max() - y.min()) < 0.05:
        return float(x.max())
    xn = (x - x.min()) / (x.max() - x.min() + 1e-9)
    yn = (y - y.min()) / (y.max() - y.min() + 1e-9)
    x1, y1, x2, y2 = xn[0], yn[0], xn[-1], yn[-1]
    d = np.abs((y2 - y1) * xn - (x2 - x1) * yn + x2 * y1 - y2 * x1) / np.hypot(y2 - y1, x2 - x1)
    return float(x[int(np.argmax(d))])


def elbow_ci(nn_dist, score, boot=BOOT):
    "Bootstrap CI of the elbow: resample kinases with replacement, recompute the elbow, take 2.5/97.5%."
    rng = np.random.default_rng(0)
    n = len(nn_dist)
    vals = []
    for _ in range(boot):
        s = rng.integers(0, n, n)                              # resample kinases with replacement
        cv = binned_curve(nn_dist[s], score[s])
        if len(cv) >= 3:
            vals.append(elbow(cv))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def tier_of(nn_dist, predictable, b_high, b_med):
    """Proximity tier from a prediction's nn_dist (distance to its nearest labelled kinase): high /
    intermediate / low proximity if predictable, else not_predicted. The predictable region [0, cut] is
    split into three equal distance bands — these are PROXIMITY tiers (raw distance), not accuracy-calibrated
    confidence. The bands are shared across methods, but each method scores a domain with ITS OWN nn_dist (a
    different labelled set), so the same domain can land in different tiers per method (e.g. PSPA-high but
    CDDM-intermediate if a closer PSPA relative exists). 'low' is the third just below the elbow."""
    nn_dist = np.asarray(nn_dist, float)
    t = np.where(nn_dist <= b_high, 'high', np.where(nn_dist <= b_med, 'intermediate', 'low'))
    return np.where(np.asarray(predictable, bool), t, 'not_predicted')


def nearest_group(feat_all, feat_col, X, grp, query_ids):
    "Group of each query domain's nearest labelled kinase (its retrieval anchor)."
    from sklearn.neighbors import NearestNeighbors
    qX = feat_all.loc[query_ids, feat_col].to_numpy(float)
    nn_idx = NearestNeighbors(n_neighbors=1).fit(X).kneighbors(qX)[1][:, 0]
    return grp[nn_idx]


def main():
    set_sns()
    feat_all = pd.read_parquet(OUT / 'kd_feat_onehot.parquet')   # one-hot for all active domains
    thr_label = 'overall flank Pearson' if THRESHOLD_METRIC == 'pearson' else 'per-position Spearman'

    cache = {}                                                   # per-target LOO scores, computed once
    for target in TARGETS:
        df, feat_col, target_col = kd_util.load_train(target, 'onehot')
        X, Y = df[feat_col].to_numpy(float), df[target_col].to_numpy(float)
        ids = df.iloc[:, 0].to_numpy()
        grp = kd_util.kinase_taxonomy(pd.Index(ids)).set_index('kinase')['group'].reindex(ids).to_numpy()
        P, nnd = loo_retrieval(X, Y)
        scores = {label: kd_util.pssm_scores(Y, P, target_col)[i] for label, i in METRICS}
        cache[target] = dict(X=X, feat_col=feat_col, grp=grp, st=grp != 'TK', nnd=nnd, scores=scores)

    # each target's OWN Kneedle elbow (shown), plus ONE conservative cutoff (THRESHOLD_TARGET's) applied to
    # all, with a bootstrap CI so the threshold is a quantified estimate rather than a single eyeballed point
    own_elbow = {t: elbow(binned_curve(cache[t]['nnd'][cache[t]['st']],
                                       cache[t]['scores'][thr_label][cache[t]['st']])) for t in TARGETS}
    c0 = cache[THRESHOLD_TARGET]
    cut = own_elbow[THRESHOLD_TARGET]
    cut_lo, cut_hi = elbow_ci(c0['nnd'][c0['st']], c0['scores'][thr_label][c0['st']])
    print(f'applied threshold = Kneedle elbow of {THRESHOLD_TARGET.upper()} {thr_label}: '
          f'nn_dist <= {cut:.2f}  [95% CI {cut_lo:.1f}-{cut_hi:.1f}]')
    print('per-target own elbows: ' + ', '.join(f'{t} {own_elbow[t]:.1f}' for t in TARGETS), flush=True)

    # proximity tiers: grade the confident predictions instead of a single hard yes/no (a broad elbow CI
    # makes a single cutoff over-precise). The predictable region [0, cut] is split into three equal
    # DISTANCE bands — PROXIMITY tiers (raw nn_dist), not accuracy-calibrated confidence. 'low' is the third
    # just below the elbow. The bands are shared, but each method scores a domain with its OWN nn_dist (a
    # different labelled set), so the same domain can tier differently per method.
    b_high, b_med = cut / 3, 2 * cut / 3
    print(f'proximity tiers (equal-distance thirds of [0, {cut:.1f}]): high nn_dist<={b_high:.1f}, '
          f'intermediate <={b_med:.1f}, low <={cut:.1f} (each method uses its own nn_dist)', flush=True)

    fig, axes = plt.subplots(len(METRICS), len(TARGETS), figsize=(6 * len(TARGETS), 4.4 * len(METRICS)),
                             squeeze=False)
    rows = []
    for j, target in enumerate(TARGETS):
        c = cache[target]
        pred = pd.read_parquet(OUT / f'kd_pred_new_{target}.parquet')   # annotate kd_07's predictions
        pred['nn_group'] = nearest_group(feat_all, c['feat_col'], c['X'], c['grp'], pred.index)
        pred['threshold'] = cut
        pred['predictable'] = pred.nn_dist <= cut
        if target in DROP_TK:
            pred.loc[pred.nn_group == 'TK', 'predictable'] = False  # TK dropped for MLP-attr (unstable)
        pred['proximity_tier'] = tier_of(pred.nn_dist.to_numpy(float), pred.predictable.to_numpy(),
                                         b_high, b_med)
        pred.to_parquet(OUT / f'kd_pred_new_{target}.parquet')
        n_pred = int(pred.predictable.sum())
        tiers = pred.loc[pred.predictable, 'proximity_tier'].value_counts().to_dict()
        rows.append({'target': target, 'threshold': round(cut, 2), 'threshold_lo': round(cut_lo, 2),
                     'threshold_hi': round(cut_hi, 2), 'own_elbow': round(own_elbow[target], 2),
                     'b_high': round(b_high, 2), 'b_med': round(b_med, 2),
                     'n_high': tiers.get('high', 0), 'n_intermediate': tiers.get('intermediate', 0),
                     'n_low': tiers.get('low', 0),
                     'n_predictable': n_pred, 'n_query': len(pred), 'drop_tk': target in DROP_TK})
        print(f'{target}: threshold nn_dist <= {cut:.2f} (from {THRESHOLD_TARGET}); own elbow '
              f'{own_elbow[target]:.1f}; predictable {n_pred:,}/{len(pred):,}'
              f'{" (TK excluded)" if target in DROP_TK else ""}', flush=True)

        for r, (label, _) in enumerate(METRICS):
            ax = axes[r][j]
            sc, cv = c['scores'][label], binned_curve(c['nnd'][c['st']], c['scores'][label][c['st']])
            ax.scatter(c['nnd'][c['st']], sc[c['st']], s=8, c='#5b8fb0', alpha=0.4, label='S/T')
            ax.scatter(c['nnd'][~c['st']], sc[~c['st']], s=10, c='#c0392b', alpha=0.5, label='TK')
            ax.plot(cv.hi, cv.s, '-o', color='#2c3e50', lw=1.5, ms=4, label='S/T binned median')
            for _, bn in cv.iterrows():                        # per-bin kinase count
                ax.annotate(f'{int(bn.n)}', (bn.hi, bn.s), textcoords='offset points', xytext=(0, 6),
                            ha='center', fontsize=6, color='#34495e')
            ax.axvspan(cut_lo, cut_hi, color='#2c3e50', alpha=0.08, lw=0,
                       label='threshold 95% CI' if (r == 0 and j == 0) else None)
            ax.axvline(cut, color='#2c3e50', ls='--', lw=1.3,
                       label=f'applied threshold {cut:.1f}' if (r == 0 and j == 0) else None)
            ax.axvline(own_elbow[target], color='#e67e22', ls=':', lw=1.4,
                       label='target own elbow' if (r == 0 and j == 0) else None)
            if r == 0:                                      # proximity-tier bands (equal-distance thirds)
                for x0, x1, col, lab in [(0, b_high, '#2ca02c', 'high proximity'),
                                         (b_high, b_med, '#e0b000', 'intermediate'),
                                         (b_med, cut, '#e67e22', 'low proximity')]:
                    ax.axvspan(x0, x1, color=col, alpha=0.10, lw=0, label=lab if j == 0 else None)
                ax.set_title(f'{target.upper()} — own elbow {own_elbow[target]:.1f}', fontsize=11)
            ax.set_xlabel('nn_dist (distance to nearest labelled kinase); labels = kinases per bin')
            ax.set_ylabel(label)
    handles, labels = axes[0][0].get_legend_handles_labels()   # one shared legend, not one per panel
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f'Proximity cutoff = Kneedle elbow (Satopää 2011) of the {THRESHOLD_TARGET.upper()} '
                 f'accuracy-vs-distance curve ({thr_label}): nn_dist <= {cut:.1f} '
                 f'[95% CI {cut_lo:.1f}-{cut_hi:.1f}, {BOOT} bootstraps], applied to all targets '
                 f'(conservative). Shaded = proximity tiers (equal-distance thirds high<={b_high:.1f} / '
                 f'intermediate<={b_med:.1f} / low<={cut:.1f}; each method tiers by its own nn_dist). '
                 f'LOO k-NN on S/T; per-bin n annotated.', fontsize=10.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_svg(FIG / 'kd_confidence_threshold.svg')
    plt.close('all')

    out = pd.DataFrame(rows)
    out.to_parquet(OUT / 'kd_confidence_threshold.parquet')
    print('\n' + out.to_string(index=False))
    print('wrote', OUT / 'kd_confidence_threshold.parquet', 'and', FIG / 'kd_confidence_threshold.svg')


if __name__ == '__main__':
    main()
