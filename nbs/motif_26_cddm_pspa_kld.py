"""motif_26 · PSPA-reference KLD sensitivity analysis for CDDM frequency.

Run from nbs: ../.venv/bin/python motif_26_cddm_pspa_kld.py
Uses the Fig. 3b cohort and motif_16 shared flank features (including s/t/y).
Groups are ordered by ascending median KLD at the main smoothing weight.
At each position, normalize both profiles on their common finite features, then
smooth each distribution as (1-alpha)*p + alpha/n_features. Average natural-log
KL(PSPA || CDDM) equally across valid positions. Main alpha=0.001; also evaluate
0.0001 and 0.01. These are uniform-mixture weights, not substrate pseudocounts.
Zero-sum positions are excluded and recorded. No signed representation enters KL.
Persist per-position and per-kinase scores before deriving summaries and plots.
Existing Spearman/AP@5 remain on motif_16's original, unsmoothed score profiles.
Outputs: out/motif_cddm_pspa_kld_*.csv, an analysis note, and fig/motif_cddm_pspa_kld*
in editable SVG/PDF plus PNG previews. Feeds the KLD columns of Supplementary Data S4.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy, spearmanr

import motif_16_compare_methods as comparison
from motif_fig3_panels import _ap_table
from katlas.utils import group_color
from kplot.utils import paper_panel, save_svg, save_pdf, set_sns
from paths import FIG, OUT

ALPHA = 0.001
ALPHAS = (0.0001, ALPHA, 0.01)
PREFIX = 'motif_cddm_pspa_kld'
KL_LABEL = 'Mean per-position KLD\n(PSPA || CDDM; nats)'
METRICS = {'spearman': 'Mean per-position Spearman', 'ap': 'AP@5'}


def position_scores(reference, observed, alphas=ALPHAS):
    """Yield auditable scores on the shared alphabet; never treat missing as zero."""
    for position in comparison._cols(reference, observed):
        p = reference[position].reindex(comparison.FEATURES).to_numpy(float)
        q = observed[position].reindex(comparison.FEATURES).to_numpy(float)
        mask = np.isfinite(p) & np.isfinite(q)
        p, q = p[mask], q[mask]
        if np.any(p < 0) or np.any(q < 0):
            raise ValueError('KLD requires nonnegative frequency profiles.')
        reason = ('too_few_features' if len(p) < comparison.MIN_RES else
                  'zero_sum' if p.sum() <= 0 or q.sum() <= 0 else '')
        if not reason:
            p, q = p / p.sum(), q / q.sum()
        for alpha in alphas:
            if not 0 < alpha < 1:
                raise ValueError('Uniform-mixture weight must be between zero and one.')
            value = (float(entropy((1-alpha)*p + alpha/len(p),
                                   (1-alpha)*q + alpha/len(q))) if not reason else np.nan)
            yield dict(position=position, alpha=alpha, kld=value,
                       n_features=len(p), exclusion=reason,
                       unsmoothed_infinite=bool(np.any((p > 0) & (q == 0))))


def calculate():
    cohort = _ap_table()
    rank = pd.read_csv(OUT / 'compare_vs_pspa_bygroup.csv')
    rank = rank[rank.method.eq('CDDM freq') & rank.kinase.isin(cohort.kinase)]
    rank = rank[['kinase', 'spearman', 'ap']].copy()
    rank['group'] = rank.kinase.map(cohort.drop_duplicates('kinase').set_index('kinase')['group'])
    pspa, cddm = comparison.load_matrix('pspa'), comparison.load_matrix('cddm')
    rows = []
    for row in rank.itertuples(index=False):
        p, q = comparison.recover(pspa.loc[row.kinase]), comparison.recover(cddm.loc[row.kinase])
        # Fail on stale rank outputs rather than correlate different source versions.
        fresh = [comparison.perpos_spearman(q, p), comparison.average_precision(q, p)]
        if not np.allclose(fresh, [row.spearman, row.ap], equal_nan=True):
            raise ValueError(f'Stale motif_16 scores for {row.kinase}; regenerate motif_16 first.')
        rows.extend(dict(kinase=row.kinase, **r) for r in position_scores(p, q))
    positions = pd.DataFrame(rows)
    positions.to_csv(OUT / f'{PREFIX}_positions.csv', index=False)
    scores = positions.groupby(['kinase', 'alpha'], as_index=False).agg(
        kld=('kld', 'mean'), n_positions=('kld', 'count'),
        n_infinite_positions=('unsmoothed_infinite', 'sum'))
    scores = scores.merge(rank, on='kinase', validate='many_to_one')
    if not np.isfinite(scores.kld).all() or (scores.kld < -1e-12).any():
        raise ValueError('Invalid aggregated KLD scores.')
    scores.to_csv(OUT / f'{PREFIX}_scores.csv', index=False)


def summarize(scores):
    base = scores[scores.alpha.eq(ALPHA)].set_index('kinase').kld
    rows = []
    for alpha, d in scores.groupby('alpha'):
        for subset in ('all', 'non-TK'):
            s = d if subset == 'all' else d[d.group.ne('TK')]
            stability = spearmanr(s.kld, s.kinase.map(base)).statistic
            for metric in METRICS:
                pair = s[['kld', metric]].dropna()
                r = spearmanr(pair.kld, pair[metric])
                rows.append(dict(alpha=alpha, subset=subset, metric=metric, n=len(pair),
                                 rho=r.statistic, pvalue=r.pvalue,
                                 rho_kld_vs_main=stability))
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / f'{PREFIX}_correlations.csv', index=False)
    scores.groupby(['alpha', 'group']).agg(
        n=('kinase', 'size'), mean_kld=('kld', 'mean'), median_kld=('kld', 'median'),
        mean_spearman=('spearman', 'mean'), mean_ap=('ap', 'mean')
    ).to_csv(OUT / f'{PREFIX}_bygroup.csv')
    return stats


def violin(ax, d, order):
    rng = np.random.default_rng(0)
    for x, group in enumerate(order):
        values = d.loc[d.group.eq(group), 'kld'].to_numpy()
        color = group_color[group]
        if len(values) > 1 and np.ptp(values) > 0:
            parts = ax.violinplot(values, positions=[x], widths=.8, showextrema=False)
            for body in parts['bodies']:
                body.set_facecolor(color)
                body.set_edgecolor('none')
                body.set_alpha(.25)
        ax.scatter(x + rng.uniform(-.18, .18, len(values)), values, s=5,
                   color=color, alpha=.65, edgecolors='none')
        ax.plot([x-.15, x+.15], [np.median(values)]*2, color=color, lw=1)
    ax.set_xticks(range(len(order)),
                  [f'{g}\n(n={sum(d.group.eq(g))})' for g in order], rotation=30, ha='right')
    ax.set_ylabel(KL_LABEL, labelpad=2)
    ax.set_ylim(bottom=0)
    ax.set_title('CDDM vs. PSPA: KLD by kinase group', pad=8)


def scatter(ax, d, metric, stats, order):
    for group in order:
        s = d[d.group.eq(group)]
        ax.scatter(s[metric], s.kld, s=7, color=group_color[group],
                   alpha=.65, edgecolors='none')
    labels = []
    for subset in ('all', 'non-TK'):
        r = stats[stats.alpha.eq(ALPHA) & stats.metric.eq(metric) & stats.subset.eq(subset)].iloc[0]
        labels.append(f'{subset}: ρ = {r.rho:.2f}, n = {r.n}')
    ax.text(.98, .98, '\n'.join(labels), transform=ax.transAxes, fontsize=6,
            ha='right', va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
    ax.set_xlabel(METRICS[metric], labelpad=2)
    ax.set_ylabel(KL_LABEL, labelpad=2)
    ax.set_ylim(bottom=0)
    title = 'KLD vs. mean per-position Spearman' if metric == 'spearman' else 'KLD vs. AP@5'
    ax.set_title(title, pad=8)


def export(fig, suffix):
    for ax in fig.axes:
        ax.tick_params(length=2, pad=1.5)
        ax.spines[['top', 'right']].set_visible(False)
    plt.figure(fig.number)
    save_svg(FIG / f'{PREFIX}{suffix}.svg')
    save_pdf(FIG / f'{PREFIX}{suffix}.pdf')
    fig.savefig(FIG / f'{PREFIX}{suffix}.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot(scores, stats, order):
    d = scores[scores.alpha.eq(ALPHA)]
    fig, ax = plt.subplots(figsize=paper_panel(1, ratio=3.2), layout='constrained')
    violin(ax, d, order)
    export(fig, '_bygroup')
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=paper_panel(1/2, ratio=1.3), layout='constrained')
        scatter(ax, d, metric, stats, order)
        export(fig, f'_vs_{metric}')
    fig = plt.figure(figsize=paper_panel(1, ratio=1.1), layout='constrained')
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    violin(axes[0], d, order)
    for ax, metric in zip(axes[1:], METRICS):
        scatter(ax, d, metric, stats, order)
    for letter, ax in zip('abc', axes):
        ax.set_title(letter, loc='left', fontweight='bold', fontsize=9)
    export(fig, '')


def main():
    set_sns()
    calculate()
    scores = pd.read_csv(OUT / f'{PREFIX}_scores.csv')
    order = (scores[scores.alpha.eq(ALPHA)].groupby('group').kld.median()
             .sort_values().index.tolist())
    stats = summarize(scores)
    plot(scores, stats, order)
    note = f'''Response Fig. R1 | PSPA-reference KLD comparison of CDDM frequency PSSMs.

Same kinase cohort as Fig. 3b (n = {scores.kinase.nunique()}).
Groups ordered by ascending median KLD at the main smoothing weight.
Shared finite flank features only, position 0 excluded; pS/pT/pY kept separate.
At each position both profiles are normalized to sum one, then smoothed with
p_smooth = (1-alpha)*p + alpha/n_features. Main alpha = {ALPHA}; checks = {ALPHAS}.
KLD = sum(p_PSPA * ln(p_PSPA/p_CDDM)), averaged equally over valid positions.
Lower KLD indicates closer distributions. This is a frequency-only comparison.
a, Violin distributions and per-kinase dots by Modi kinase group; short lines, medians.
b, c, KLD versus original mean per-position Spearman and AP@5, respectively.
Colors in b/c match the labeled groups in a. Annotations: Spearman rho across kinases,
computed for all kinases and separately for non-TK kinases. No fitted line is shown.
Existing rank metrics are checked against current raw inputs and are not smoothed.
CSV outputs retain positional exclusions, unsmoothed infinity flags, per-kinase scores,
group summaries, and correlations at every smoothing weight. Smoothing is not
count-based estimation; absolute KLD depends on it. Correlations are descriptive.

{stats.to_string(index=False)}
'''
    (OUT / f'{PREFIX}_notes.txt').write_text(note)
    print(note)


if __name__ == '__main__':
    main()
