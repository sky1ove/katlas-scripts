"""kd_04d · Model x feature comparison figures + tables, from the kd_04b grid AND the kd_04c DNN scores.

Pure analysis/plotting — no training, no re-scoring. Concatenates the per-kinase scores from the
sklearn grid (kd_04b) and the proper DNN baseline (kd_04c, MLP on the distribution targets) — both
persisted in the same schema (one row per model x feature x **repeat** x kinase) — and draws, per
target, two views:
  heatmap  models x features, one panel per metric (Spearman, AP@5, Pearson), median over kinases and
           repeats — the whole grid at a glance.
  bars     grouped bars, models (ranked by CV Spearman) x features; error bars = std **across the
           REPEAT_SEEDS repeats** (how much the number moves under a different fold shuffle), and a
           dashed line marks the selected kNN×onehot cell so it is clear which cells clear it.
The selected (model, feature) cell (kd_07: k-NN on one-hot) is boxed in the heatmap and lined in the bars.

There is no held-out test set: kNN was chosen for deployment reasons, not by topping this grid, so the
grid is descriptive and every cell is an out-of-fold CV score over the target's whole labelled set,
repeated to damp partition noise (see kd_04b). "Is a cell really better than the selected kNN×onehot"
is answered by a **two-sided Wilcoxon signed-rank test** across the matched per-kinase OOF scores (S/T
only), with the rank-biserial effect size and a Benjamini–Hochberg FDR adjustment across the grid cells,
reported in the workbook — not by whether the error bars overlap. The grid is descriptive (kNN is
prespecified for deployment), so the FDR column is for completeness, not a selection rule.

Each figure is drawn twice: over **all** kinases, and over **ST kinases only** (`_ST`, tyrosine
kinases dropped — TK flanking motifs are weak and muddy the comparison), sliced on the `group` column.

Inputs   out/kd_grid_scores.parquet (kd_04b), out/kd_dnn_scores.parquet (kd_04c, optional)
Outputs  fig/kd_model_selection_{pspa,cddm,mlp_attr}{,_ST}.svg (heatmaps),
         fig/kd_model_selection_bar_{pspa,cddm,mlp_attr}{,_ST}.svg (grouped bars),
         out/kd_model_selection.xlsx (one sheet per target x subset {ST,TK,all}: features x models
         matrices of each metric + the paired-vs-kNN×onehot Δ and Wilcoxon p)

Run:  python nbs/kd_04d_grid_figures.py   (run kd_04b and kd_04c first)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from paths import FIG, OUT

from kplot.utils import save_svg, set_sns

# mirror kd_04b's grid axes (kept local so this figures script has no modelling dependency)
FEATURES = ['onehot', 'onehot_pca', 'esm', 't5']
TARGETS = ['pspa', 'cddm', 'mlp_attr']       # two distributions + the signed MLP-attribution target
METRICS = ['spearman', 'ap', 'pearson']
METRIC_TITLE = {'spearman': 'per-position Spearman', 'ap': 'AP@5', 'pearson': 'overall Pearson'}
SHIPPED = ('kNN', 'onehot')     # the deployed (model, feature) (kd_07) — boxed / lined / paired against
#: non-competitive references pinned to the bottom, in this exact order (mean last — it is the floor,
#: not a model, so it sits dead last)
BASELINES = ('SVR', 'Linear', 'mean')


def load_scores():
    "The sklearn grid (kd_04b) + the DNN baseline (kd_04c), concatenated — same per-kinase schema."
    grid = OUT / 'kd_grid_scores.parquet'
    if not grid.exists():
        sys.exit(f'{grid} not found - run kd_04b_model_comparison.py first')
    frames = [pd.read_parquet(grid)]
    dnn = OUT / 'kd_dnn_scores.parquet'
    if dnn.exists():
        frames.append(pd.read_parquet(dnn))                  # MLP on the distribution targets (kd_04c)
    else:
        print('  note: kd_dnn_scores.parquet not found - run kd_04c_dnn.py to include the MLP')
    return pd.concat(frames, ignore_index=True)


def subsets(scores):
    "The kinase subsets to draw: all kinases, and ST only (TK dropped — tyrosine motifs are weak)."
    st = scores[scores['group'] != 'TK']                     # `group` column persisted upstream
    return {'': ('', scores),
            '_ST': (' — ST kinases only (TK excluded)', st)}


def model_order(scores):
    "One global order for every figure: contenders (by mean ST CV Spearman, best first), then the\
 baseline references pinned to the bottom — so a model sits in the same row in every panel."
    cv = scores[scores['group'] != 'TK']
    rank = cv.groupby('model')['spearman'].mean().sort_values(ascending=False).index.tolist()
    return [m for m in rank if m not in BASELINES] + [m for m in BASELINES if m in rank]  # fixed tail


def _cell_medians(scores, label, metric, order):
    "features x models -> median over kinases and repeats, as a (model x feature) matrix in `order`."
    sub = scores[scores.target == label]
    return (sub.groupby(['model', 'feature'])[metric].median()
            .unstack('feature').reindex(index=order, columns=FEATURES))


def heatmap_figure(scores, label, order, suffix='', note=''):
    "Model x feature heatmaps, one panel per metric — median over kinases and repeats."
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.6 * len(METRICS), 0.34 * len(order) + 2.2),
                             squeeze=False)
    for c, metric in enumerate(METRICS):
        ax = axes[0, c]
        grid = _cell_medians(scores, label, metric, order)
        arr = grid.to_numpy()
        im = ax.imshow(arr, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=30, ha='right')
        ax.set_yticks(range(len(order)), order if c == 0 else [''] * len(order), fontsize=8)
        mid = np.nanmean(arr)
        for i in range(len(order)):
            for j in range(len(FEATURES)):
                v = arr[i, j]
                ax.text(j, i, '' if np.isnan(v) else f'{v:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if v < mid else 'black')
        if SHIPPED[0] in order:
            ax.add_patch(plt.Rectangle((FEATURES.index(SHIPPED[1]) - 0.5, order.index(SHIPPED[0]) - 0.5),
                                       1, 1, fill=False, edgecolor='#c0392b', lw=2))
        ax.set_title(METRIC_TITLE[metric], fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'Model x feature grid on {label.upper()} — repeated subfamily-grouped CV, flank only '
                 f'(median; selected {SHIPPED[0]} x {SHIPPED[1]}, boxed){note}', fontsize=11)
    fig.tight_layout()
    save_svg(FIG / f'kd_model_selection_{label}{suffix}.svg')
    plt.close('all')
    print('  wrote', FIG / f'kd_model_selection_{label}{suffix}.svg')


def bar_figure(scores, label, order, suffix='', note=''):
    "Grouped bars — per-position Spearman by model (ranked) x feature; error bars = std across repeats."
    sub = scores[scores.target == label]
    # one value per (model, feature, repeat) = mean over kinases; barplot's 'sd' is then across repeats
    rep = sub.groupby(['model', 'feature', 'repeat'])['spearman'].mean().reset_index()
    ship = rep[(rep.model == SHIPPED[0]) & (rep.feature == SHIPPED[1])]['spearman'].mean()

    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(order) + 1.8))
    sns.barplot(data=rep, y='model', x='spearman', hue='feature', order=order, hue_order=FEATURES,
                ax=ax, errorbar='sd', err_kws={'lw': 0.8}, capsize=0.25)
    if not np.isnan(ship):
        ax.axvline(ship, color='#c0392b', ls='--', lw=1.2,
                   label=f'selected {SHIPPED[0]}x{SHIPPED[1]} ({ship:.2f})')
    ax.set_xlabel('per-position Spearman (flank) — mean ± sd across repeats')
    ax.set_ylabel('')
    ax.legend(title='feature', fontsize=7, title_fontsize=8, loc='lower right')
    fig.suptitle(f'Model x feature on {label.upper()} — repeated subfamily-grouped CV, flank only{note}',
                 fontsize=11)
    fig.tight_layout()
    save_svg(FIG / f'kd_model_selection_bar_{label}{suffix}.svg')
    plt.close('all')
    print('  wrote', FIG / f'kd_model_selection_bar_{label}{suffix}.svg')


def paired_vs_shipped(sub):
    """Per (model, feature): a two-sided Wilcoxon signed-rank test of the repeat-averaged per-kinase OOF
    Spearman against the shipped kNN×onehot cell (same kinases). Returns matrices of the median Δ, the
    rank-biserial effect size r, the exact raw p, and a Benjamini–Hochberg FDR-adjusted p across the grid
    cells. The grid is DESCRIPTIVE — kNN is prespecified for deployment, not selected on these p — so the
    FDR column is reported for completeness, not as a selection rule."""
    import stats_util as st
    km = sub.groupby(['model', 'feature', 'kd_ID'])['spearman'].mean()            # avg over repeats
    ship = km.loc[SHIPPED[0], SHIPPED[1]] if SHIPPED[0] in km.index.get_level_values(0) else None
    delta, rbis, pval = {}, {}, {}
    for (m, f), s in km.groupby(level=['model', 'feature']):
        s = s.droplevel(['model', 'feature'])
        if ship is None:
            delta[(m, f)] = rbis[(m, f)] = pval[(m, f)] = np.nan
            continue
        common = s.index.intersection(ship.index)
        a, b = s.reindex(common).to_numpy(), ship.reindex(common).to_numpy()
        if (m, f) == SHIPPED or len(common) < 5:
            delta[(m, f)] = float(np.nanmedian(a - b)); rbis[(m, f)] = pval[(m, f)] = np.nan
            continue
        rep = st.wilcoxon_report(a, b, ci=False)                                  # W/r/p only (no CI here)
        delta[(m, f)], rbis[(m, f)], pval[(m, f)] = rep['median_delta'], rep['r'], rep['p']
    p_mat = pd.Series(pval).unstack()
    vals = p_mat.to_numpy()                                                       # BH-FDR across all cells
    q_mat = pd.DataFrame(st.fdr_bh(vals.ravel()).reshape(vals.shape),
                         index=p_mat.index, columns=p_mat.columns)
    return pd.Series(delta).unstack(), pd.Series(rbis).unstack(), p_mat, q_mat


def sheet_frame(sub, order):
    "One sheet: metric matrices (median over kinases+repeats) + paired-vs-kNN×onehot Δ, r, and exact p (raw + BH-FDR)."
    order = [m for m in order if m in set(sub.model)]                        # global order, present rows
    cols = ['model'] + FEATURES

    def block(title, mat, as_p=False):
        mat = mat.reindex(index=order, columns=FEATURES)
        # p-values are formatted in scientific notation (exact, never rounded to 0); other cells to 3 dp
        mat = mat.apply(lambda c: c.map(lambda v: '' if pd.isna(v) else f'{v:.1e}')) if as_p else mat.round(3)
        mat = mat.reset_index().rename(columns={'index': 'model'})
        head = pd.DataFrame([[title] + [''] * len(FEATURES)], columns=cols)
        blank = pd.DataFrame([[''] * len(cols)], columns=cols)
        return [head, mat[cols], blank]

    parts = []
    for metric in METRICS:
        mat = sub.groupby(['model', 'feature'])[metric].median().unstack('feature')
        parts += block(f'{metric} (median over kinases + repeats)', mat)
    delta, rbis, pval, qval = paired_vs_shipped(sub)
    ship = f'{SHIPPED[0]}x{SHIPPED[1]}'
    parts += block(f'paired Δ Spearman vs {ship} (repeat-avg per kinase)', delta)
    parts += block(f'rank-biserial r vs {ship}', rbis)
    parts += block(f'two-sided Wilcoxon p vs {ship} (exact)', pval, as_p=True)
    parts += block(f'BH-FDR p vs {ship} (across grid cells)', qval, as_p=True)
    return pd.concat(parts, ignore_index=True)[cols]


def selected_cell_summary(scores):
    """The shipped kNN×onehot cell's headline metrics with a 95% CI, per target × subset (ST / all).

    The grid is descriptive, but the *selected* model should still report an interval, not a bare point.
    The point is the grid's own convention – the median over all (kinase × repeat) rows, i.e. exactly the
    heatmap / sheet cell – and the CI is a CLUSTER BOOTSTRAP over kinases (resample kinases, pool their
    rows, take the median). Point and CI therefore use the same aggregation, and the CI still resamples at
    the biological unit (the same one the paired-vs-shipped Wilcoxon uses)."""
    from stats_util import boot_ci_cluster
    rows = []
    for target in TARGETS:
        for sname, mask in [('ST', scores['group'] != 'TK'), ('all', pd.Series(True, index=scores.index))]:
            sub = scores[(scores.target == target) & mask &
                         (scores.model == SHIPPED[0]) & (scores.feature == SHIPPED[1])]
            if sub.empty:
                continue
            row = {'target': target, 'subset': sname, 'model': f'{SHIPPED[0]}×{SHIPPED[1]}',
                   'n_kinases': int(sub['kd_ID'].nunique())}
            for mt in METRICS:
                lo, hi = boot_ci_cluster(sub[mt].to_numpy(), sub['kd_ID'].to_numpy(), np.median)
                row[f'{mt}_median'] = round(float(np.nanmedian(sub[mt])), 3)    # median over all rows = grid cell
                row[f'{mt}_95CI'] = f'[{lo:.2f}, {hi:.2f}]'
            rows.append(row)
    return pd.DataFrame(rows)


def tables(scores, order):
    "Excel workbook: one sheet per (target, subset ST/TK/all), features x models matrices + paired test."
    path = OUT / 'kd_model_selection.xlsx'
    subsets_tab = [('ST', scores['group'] != 'TK'),                # non-tyrosine (the selection focus)
                   ('TK', scores['group'] == 'TK'),
                   ('all', pd.Series(True, index=scores.index))]
    sel = selected_cell_summary(scores)
    with pd.ExcelWriter(path, engine='openpyxl') as xl:
        sel.to_excel(xl, sheet_name='selected_cell', index=False)       # shipped model + 95% CI, up front
        for target in TARGETS:
            for sname, mask in subsets_tab:
                sub = scores[(scores.target == target) & mask]
                if sub.empty:
                    continue
                sheet_frame(sub, order).to_excel(xl, sheet_name=f'{target}_{sname}'[:31], index=False)
    print('  wrote', path)
    print('  selected kNN×onehot cell (median [95% CI] over kinases):')
    print(sel.to_string(index=False))


def main():
    set_sns()
    scores = load_scores()
    order = model_order(scores)                                    # one global order shared by all figures
    for suffix, (note, sub) in subsets(scores).items():
        for label in TARGETS:
            print(f"\n== {label.upper()}{note or ' — all kinases'} ==")
            heatmap_figure(sub, label, order, suffix, note)
            bar_figure(sub, label, order, suffix, note)
    tables(scores, order)


if __name__ == '__main__':
    main()
