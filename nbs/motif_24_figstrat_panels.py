"""motif_24_figstrat · Paper Fig. 4 panels a and b: per-kinase motif sharpness (spec_index) and how
CDDM-vs-PSPA agreement scales with it. Panels c-f of Fig. 4 (CK1E / GSK3B / ERK1 / ABL1 k-means
sub-motifs) come from motif_25_stratify_pspa + the per-kinase logo/heatmap plotters; this script builds
Fig. 4 a and b.

  a  spec_index (sharpness) by kinase group for PSPA / CDDM / MLP-attr, grouped bars (group x method
     mean) with SEM error bars. Groups ordered by overall mean sharpness.
  b  per-kinase CDDM-vs-PSPA(scale) overall flank Pearson against the kinase's PSPA spec_index: sharper
     kinases (high spec_index) agree more between the two methods.

Sized via kplot.utils.paper_panel; method colors follow the scoring palette (PSPA blue, CDDM green,
MLP-attr orange). Panel b's per-kinase values are persisted so the point estimate is a derive, not a
re-score.

Inputs   out/specificity_metrics.parquet (motif_09b); kdata: cddm, pspa_scale, kinase_info;
         motif_16.recover (shared-alphabet PSSM recovery)
Outputs  out/strat_agreement_vs_spec.csv; fig/figstrat_a_spec_bar.svg, fig/figstrat_b_{ap,pearson}_vs_spec.svg

Run:  python nbs/motif_24_figstrat_panels.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from motif_16_compare_methods import average_precision, recover
from paths import FIG, OUT

import kdata
from katlas.utils import group_color
from kplot.bar import plot_group_violin
from kplot.utils import paper_panel, save_svg, set_sns

METHOD_COL = {'PSPA': '#4292c6', 'CDDM': '#41ab5d', 'MLP-attr': '#f16913'}
SPEC = OUT / 'specificity_metrics.parquet'


def _spec():
    return pd.read_parquet(SPEC)


# ----------------------------------------------------------- Fig 4a (panel_a): sharpness by group -----
def panel_a(frac=2 / 3, ratio=2.8, style='bar'):
    """spec_index by group, PSPA / CDDM / MLP-attr. `style='bar'` = grouped bars (group x method mean)
    with SEM error bars; `style='violin'` = grouped violins split by method + per-kinase dots
    (kplot.bar.plot_group_violin). Groups ordered by overall mean sharpness; zero line kept for both."""
    d = _spec()
    order = d.groupby('group').spec_index.mean().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=paper_panel(frac, ratio=ratio))
    if style == 'bar':
        sns.barplot(data=d, x='group', y='spec_index', hue='method', order=order,
                    hue_order=['PSPA', 'CDDM', 'MLP-attr'], palette=METHOD_COL, ax=ax,
                    errorbar='se', capsize=.15, err_kws={'linewidth': .6, 'color': '0.3'},
                    edgecolor='black', linewidth=.4, gap=.12)   # black edges + gap so adjacent methods separate
    else:
        plot_group_violin(d, 'spec_index', 'group', hue='method', palette=METHOD_COL,
                          hue_order=['PSPA', 'CDDM', 'MLP-attr'], order=order, ax=ax,
                          dot_size=1.5, violin_alpha=.30, legend=False)
    ax.axhline(0, color='0.7', lw=.5, zorder=0)
    ax.set_ylabel('spec_index (sharpness)', labelpad=2)
    ax.set_xlabel('')
    ax.tick_params(length=2, pad=1.5)
    ax.set_xticklabels(order, rotation=35, ha='right')
    ax.legend(fontsize=6, frameon=False, ncol=3, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    save_svg(FIG / f'figstrat_a_spec_{style}.svg')
    plt.close(fig)


# -------------------------------------------------------- Fig 4b (panel_b): agreement vs sharpness -
def _agreement_vs_spec():
    """Per kinase: CDDM-vs-PSPA(scale) agreement by two metrics, plus the kinase's PSPA spec_index.
    - flank Pearson: whole-flank magnitude agreement (rises with sharpness partly by construction).
    - AP@5: does CDDM recover PSPA's 5 strongest flank cells; rank-based, so much less magnitude-driven.
    """
    cddm = kdata.load('cddm')
    pspa = kdata.load('pspa_scale')
    pspa = pspa[~pspa.index.str.contains('_TYR')]
    ps = _spec().query('method == "PSPA"').set_index('kinase').spec_index
    g = kdata.load('kinase_info').drop_duplicates('kinase').set_index('kinase').group

    def flank_pearson(a, b):
        cols = [c for c in a.columns if isinstance(c, (int, np.integer)) and c != 0 and c in b.columns]
        av, bv = a[cols].stack(), b[cols].stack()
        idx = av.index.intersection(bv.index)
        return float(np.corrcoef(av[idx].values, bv[idx].values)[0, 1]) if len(idx) > 2 else np.nan

    rows = []
    for k in sorted(set(cddm.index) & set(pspa.index) & set(ps.index)):
        a, b = recover(cddm.loc[k]), recover(pspa.loc[k])            # shared alphabet (PSPA pS==pT)
        rows.append((k, ps[k], flank_pearson(a, b), average_precision(a, b), g.get(k)))
    df = pd.DataFrame(rows, columns=['kinase', 'pspa_spec', 'pearson', 'ap', 'group'])
    return df.dropna(subset=['pspa_spec', 'group'])


#: (column, y-label, output filename) per agreement metric
_METRIC = {'pearson': ('pearson', 'CDDM–PSPA flank Pearson', 'figstrat_b_pearson_vs_spec.svg'),
           'ap': ('ap', 'CDDM–PSPA AP@5', 'figstrat_b_ap_vs_spec.svg')}


def panel_b(metric='ap', frac=1 / 3, ratio=1.15, label_spec_min=1.7, legend=True):
    """Per-kinase CDDM-PSPA agreement (`metric` = 'ap' or 'pearson') vs the kinase's PSPA spec_index.
    Only the sharpest kinases (spec_index > `label_spec_min`) are labeled (adjustText): the far-right
    cluster, i.e. the CK1 family and GSK3, whose motifs are extremely peaked. No fit line or divider:
    the read is the spread, the Spearman correlation, and where the sharpest kinases land. `legend`
    adds a boxed kinase-group color key outside the axes on the right."""
    from adjustText import adjust_text
    from scipy.stats import spearmanr
    df = _agreement_vs_spec()
    df.round(4).to_csv(OUT / 'strat_agreement_vs_spec.csv', index=False)
    col, ylabel, fname = _METRIC[metric]
    d = df.dropna(subset=[col])
    groups = [g for g in group_color if g in set(d.group)]
    fig, ax = plt.subplots(figsize=paper_panel(frac, ratio=ratio))
    for g in groups:
        sub = d[d.group == g]
        ax.scatter(sub.pspa_spec, sub[col], s=6, color=group_color[g], edgecolor='none', alpha=.8)
    tip = d[d.pspa_spec > label_spec_min]                            # the sharpest kinases (far right)
    ax.margins(x=.16, y=.10)                                         # room so right-edge labels sit outside
    texts = [ax.text(r.pspa_spec, r[col], r.kinase, fontsize=6, color='0.15', ha='center', va='center')
             for _, r in tip.iterrows()]
    adjust_text(texts, ax=ax, expand=(1.25, 1.4), force_text=(.25, .4),   # just stagger the labels clear;
                arrowprops=dict(arrowstyle='-', color='0.55', lw=.4), min_arrow_len=18)  # leader only if moved far
    res = spearmanr(d.pspa_spec, d[col])                            # sharper motifs are easier to recover
    ax.text(.03, .97, f'Spearman \u03c1 = {res.correlation:.2f}\np = {res.pvalue:.2e}',
            transform=ax.transAxes, fontsize=6, va='top')            # n goes in the legend, not the panel
    ax.set_xlabel('PSPA spec_index (sharpness)', labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.tick_params(length=2, pad=1.5)
    ax.spines[['top', 'right']].set_visible(False)
    if legend:                                                       # group color key, boxed, outside right
        handles = [Line2D([0], [0], marker='o', ls='', ms=3, mec='none', mfc=group_color[g], label=g)
                   for g in groups]
        ax.legend(handles=handles, fontsize=6, frameon=True, loc='center left',
                  bbox_to_anchor=(1.0, 0.5), handletextpad=.3, labelspacing=.25, borderpad=.35)
    save_svg(FIG / fname)
    plt.close(fig)
    print(f'panel b ({metric}): rho={res.correlation:.2f} p={res.pvalue:.2e} n={len(d)}; '
          f'labeled {len(tip)} sharpest (spec_index > {label_spec_min}): '
          f'{", ".join(tip.sort_values("pspa_spec", ascending=False).kinase)} (n = {len(d)})')


def main():
    set_sns()
    panel_a()
    panel_b('ap')          # rank-based, magnitude-robust (preferred)
    panel_b('pearson')     # flank-magnitude version, for comparison
    print('wrote figstrat_a_spec_bar.svg + figstrat_b_{ap,pearson}_vs_spec.svg')


if __name__ == '__main__':
    main()
