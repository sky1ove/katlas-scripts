"""scoring_fig2_panels · Build the Figure 2 panels for the paper (kinase-for-a-site benchmark).

Each panel is sized with `kplot.utils.paper_panel(frac, ratio)` to the fixed 180 x 170 mm page, 7 pt
fonts, 8 pt panel titles, 0.6 pt frame, tight ticks, SVG (place at 100%). This is the hand-built
paper-figure generator (house look, like kd_fig7_panels / motif_fig3_panels), distinct from
scoring_04d_figures which draws the large exploratory/HTML figures. Method colors follow the scoring
palette (PSPA blue, CDDM green, MLP orange). Data + the window-curve aggregation are reused from
scoring_04d_figures (single source of truth for the window map and macro-by-window helper).

Panels (all six built by `build(nk_cap, prefix)`)
  a, b  macro recall@10 vs flank window, ST / Tyr        (55.8 x 38.5 mm, ratio 1.45)
  c, d  nine-method overall bars, ST / TK                (120 mm wide, ratio 3.15)
  e, f  per-kinase-group micro / macro recall@10 bars    (180 mm wide, ratio 4.25)
  + a shared method-color legend (fig2_method_legend.svg)

The main figure uses the reliably annotated num_kin <= 10 test subset (`su.NK_MAIN`). The supplement
(scoring_fig2supp_panels.py) calls `build(None, 'fig2supp')` for the same panels on every test site.

Inputs   out/scoring_pairs/{pspa,cddm,cddm_seq}_pairs.parquet, window_{mlp,cddm,pspa}_pairs.parquet,
         out/scoring_split.parquet, out/scoring_pool.parquet
Outputs  fig/fig2{a..f}_*.svg, fig/fig2_method_legend.svg

Run:  python nbs/scoring_fig2_panels.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import scoring_04d_figures as f4
import scoring_util as su
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from paths import FIG

from kplot.utils import paper_panel, save_svg, set_sns


def _macro_ci(vals, B=su.N_BOOT, seed=0):
    "Bootstrap 95% CI for macro recall@10: resample KINASES, re-average their per-kinase recall."
    v = np.asarray(vals, float)
    if len(v) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

#: short legend labels (drop the parenthetical descriptor: "PSPA", not "PSPA (experimental)")
LABEL = {k: k.split(' (')[0] for k in f4.WIN}
#: c/d metric columns (source column, display title); AUCDF first, then the two recall@10 views
OVMETRICS = [('AUCDF', 'AUCDF'), ('micro', 'micro recall@10'), ('macro', 'macro recall@10')]


def _cap(df, nk_cap):
    "Restrict to sites annotated by <= nk_cap kinases; nk_cap=None keeps every test site (the supplement)."
    return df if nk_cap is None else df[df.num_kin <= nk_cap]


def macro_by_window_ci(fname, split, wmap, metric, nk_cap):
    "Per (window, branch): macro recall@10 (mean over kinases) with a bootstrap 95% CI over kinases."
    p = pd.read_parquet(su.RES / fname)
    p = _cap(p[p.split == split], nk_cap).copy()
    p['w'] = p.window.map(wmap) if wmap else p.window.astype(int)
    unmapped = sorted(p.loc[p.w.isna(), 'window'].unique())
    assert not unmapped, f'{fname}: window labels missing from the map: {unmapped}'
    perk = su.summarize(p, ['w', 'branch', 'kinase'])            # per-kinase recall@10 at each window
    rows = []
    for (w, br), g in perk.groupby(['w', 'branch'], observed=True):
        kv = g[metric].to_numpy()
        lo, hi = _macro_ci(kv, seed=int(w))                     # vary seed by window
        rows.append({'w': w, 'branch': br, 'mean': float(kv.mean()), 'lo': lo, 'hi': hi})
    return pd.DataFrame(rows).set_index(['w', 'branch'])


def load_methods(nk_cap):
    "The seven scored methods renamed for display, at a given num_kin cap (None = all test sites)."
    raw = pd.concat([pd.read_parquet(su.RES / f) for f in
                     ['pspa_pairs.parquet', 'cddm_pairs.parquet', 'cddm_seq_pairs.parquet']],
                    ignore_index=True)
    missing = sorted(set(f4.NAME) - set(raw.method))
    assert not missing, f'missing methods: {missing}'
    pairs = raw[raw.method.isin(f4.NAME)].copy()
    pairs['method'] = pairs.method.map(f4.NAME)
    return _cap(pairs, nk_cap).copy()


def window_panel(pools, br, fname, nk_cap, split='test', frac=(50 / 1.3 * 1.45) / 180, ratio=1.45,
                 metric='top10', mlabel='recall@10'):   # height fixed at 50/1.3 mm (~38.5), width ~55.8 mm
    "One paper panel: macro recall@10 vs flank window; shaded band = bootstrap 95% CI over kinases."
    curves = {k: macro_by_window_ci(v[0], split, v[1], metric, nk_cap) for k, v in f4.WIN.items()}
    xticks = sorted({int(x) for c in curves.values() for x in c.index.get_level_values('w')})

    fig, ax = plt.subplots(figsize=paper_panel(frac, ratio=ratio))
    for name, (_, _, col, mk) in f4.WIN.items():
        d = curves[name].xs(br, level='branch').sort_index()
        x = d.index.to_numpy()
        ax.fill_between(x, d['lo'], d['hi'], color=col, alpha=0.15, linewidth=0)   # 95% CI over kinases
        ax.plot(x, d['mean'], mk, color=col, ms=3, lw=1.0, label=LABEL[name])
    ax.set_xticks(xticks)
    chance = 10 / len(pools[br])
    ax.axhline(chance, ls='--', color='0.6', lw=0.6)
    ax.text(0.38, chance, 'random', fontsize=7, color='0.6', ha='center', va='bottom',
            transform=ax.get_yaxis_transform())                  # left of the lower-right legend
    ax.set_xlabel('window half-width (±w)', labelpad=1.5)
    ax.set_ylabel(f'macro {mlabel}', labelpad=1.5)
    ax.tick_params(length=2, pad=1.5)
    ax.grid(alpha=0.3, lw=0.4)
    hd = dict(zip(*reversed(ax.get_legend_handles_labels())))     # label -> handle
    order = [LABEL['PSPA (experimental)'], LABEL['CDDM PSSM (generative)'],   # PSPA first
             LABEL['CDDM-seq MLP (discriminative)']]
    ax.legend([hd[l] for l in order], order, loc='lower right', ncol=1, frameon=False,
              handlelength=1.4, handletextpad=0.4, labelspacing=0.3, borderpad=0.3)  # one entry per row
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.4)
    save_svg(fname)
    plt.close('all')
    print('  wrote', fname)


def load_pairs_full(nk_cap):
    "The nine rows (Random/Dummy + seven methods) as per-pair scores at a given num_kin cap."
    split = su.load_split()
    pools, _ = su.load_pool()
    pairs = load_methods(nk_cap)
    return pd.concat([pairs,
                      f4.baseline_pairs(pairs, split, pools, 'Random'),
                      f4.baseline_pairs(pairs, split, pools, 'Dummy')], ignore_index=True)


def overall_stats(pairs):
    "Per (branch, method): mean + kinase-bootstrap 95% CI for AUCDF, micro and macro recall@10."
    perk = su.summarize(pairs, ['method', 'branch', 'kinase'])
    rows = []
    for br in ['ST', 'Tyr']:
        d = pairs[pairs.branch == br]
        agg = su.summarize(d, ['method', 'branch']).set_index('method')    # pooled (micro) points
        for m in f4.ALL:
            ci = su.boot_micro_ci(d[d.method == m])                        # kinase cluster bootstrap
            kv = perk[(perk.method == m) & (perk.branch == br)]['top10'].to_numpy()
            mlo, mhi = _macro_ci(kv)
            rows.append({'branch': br, 'method': m,
                         'AUCDF': agg.loc[m, 'AUCDF'], 'AUCDF_lo': ci['AUCDF'][0], 'AUCDF_hi': ci['AUCDF'][1],
                         'micro': agg.loc[m, 'top10'], 'micro_lo': ci['top10'][0], 'micro_hi': ci['top10'][1],
                         'macro': float(kv.mean()), 'macro_lo': mlo, 'macro_hi': mhi})
    return pd.DataFrame(rows)


def _xerr(d, key):
    "Asymmetric [below, above] error lengths from the *_lo / *_hi CI columns (NaN CIs -> 0 length)."
    m = d[key].to_numpy(float)
    e = np.vstack([m - d[f'{key}_lo'].to_numpy(float), d[f'{key}_hi'].to_numpy(float) - m])
    return np.clip(np.nan_to_num(e, nan=0.0), 0, None)


def overall_panel(stats, branch, blabel, fname, frac=120 / 180, ratio=3.15):
    "One paper panel: nine methods on y, three metric columns; error bars = kinase-bootstrap 95% CI."
    d = stats[stats.branch == branch].set_index('method').reindex(f4.ALL)
    y = np.arange(len(f4.ALL))[::-1]                              # methods[0] (Random) at the top
    colors = [f4.PALETTE[m] for m in f4.ALL]
    fig, axes = plt.subplots(1, len(OVMETRICS), figsize=paper_panel(frac, ratio=ratio),
                             squeeze=False, sharey=True)
    for ci, (mk, ml) in enumerate(OVMETRICS):
        ax = axes[0][ci]
        ax.barh(y, d[mk].to_numpy(float), color=colors, height=0.72)
        ax.errorbar(d[mk].to_numpy(float), y, xerr=_xerr(d, mk), fmt='none',
                    ecolor='0.45', elinewidth=0.7, capsize=0)          # capless thin gray CI (clean)
        ax.set_title(f'{blabel}: {ml}', fontsize=8)          # branch prefix; colon (no dash-as-punctuation)
        if mk == 'AUCDF':
            ax.set_xlim(0.45, None)
        ax.tick_params(length=2, pad=1.5)
        ax.grid(axis='x', alpha=0.3, lw=0.4)
    axes[0][0].set_yticks(y)                                  # label once on the shared y-axis; sharey
    axes[0][0].set_yticklabels(f4.ALL)                        # hides the tick labels on the other columns
    sns.despine(fig)
    plt.tight_layout(pad=0.4, w_pad=1.8)                          # gap so adjacent x-tick labels don't touch
    save_svg(fname)
    plt.close('all')
    print('  wrote', fname)


def group_stats(pairs, macro):
    "Per (kinase group, method): mean + kinase-bootstrap 95% CI for micro (pooled) or macro recall@10."
    rows = []
    if macro:
        perk = su.summarize(pairs, ['method', 'kinase_group', 'kinase'])
        for (m, g), sub in perk.groupby(['method', 'kinase_group'], observed=True):
            kv = sub['top10'].to_numpy()
            lo, hi = _macro_ci(kv)
            rows.append({'kinase_group': g, 'method': m, 'mean': float(kv.mean()), 'lo': lo, 'hi': hi})
    else:
        agg = su.summarize(pairs, ['method', 'kinase_group']).set_index(['method', 'kinase_group'])
        for (m, g), sub in pairs.groupby(['method', 'kinase_group'], observed=True):
            lo, hi = su.boot_micro_ci(sub).get('top10', (np.nan, np.nan))
            rows.append({'kinase_group': g, 'method': m, 'mean': agg.loc[(m, g), 'top10'], 'lo': lo, 'hi': hi})
    return pd.DataFrame(rows)


def group_panel(stats, metric, fname, frac=1.0, ratio=4.25):         # 180 mm wide, ~42 mm tall
    "One full-width paper panel: recall@10 per kinase group, nine dodged bars; kinase-bootstrap 95% CI."
    gorder = [g for g in f4.GORDER if g in stats.kinase_group.unique()]
    nm = len(f4.ALL)
    w = 0.8 / nm
    S = stats.set_index(['kinase_group', 'method'])
    fig, ax = plt.subplots(figsize=paper_panel(frac, ratio=ratio))
    for j, m in enumerate(f4.ALL):
        xs, ys, elo, ehi = [], [], [], []
        for gi, g in enumerate(gorder):
            if (g, m) in S.index:
                r = S.loc[(g, m)]
                xs.append(gi + (j - (nm - 1) / 2) * w)
                ys.append(r['mean'])
                elo.append(0.0 if np.isnan(r['lo']) else r['mean'] - r['lo'])
                ehi.append(0.0 if np.isnan(r['hi']) else r['hi'] - r['mean'])
        ax.bar(xs, ys, width=w, color=f4.PALETTE[m], linewidth=0)
        ax.errorbar(xs, ys, yerr=np.vstack([elo, ehi]), fmt='none',
                    ecolor='0.45', elinewidth=0.5, capsize=0)          # capless thin gray CI (clean)
    ax.set_xticks(range(len(gorder)))
    ax.set_xticklabels(gorder)
    ax.set_xlim(-0.5, len(gorder) - 0.5)
    ax.set_ylim(0, None)
    ax.set_xlabel('')                                            # group names are self-evident on the axis
    ax.set_ylabel(metric, labelpad=1.5)                          # micro/macro recall@10 on the y-axis
    ax.tick_params(length=2, pad=1.5)
    ax.grid(axis='y', alpha=0.3, lw=0.4)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.4)
    save_svg(fname)
    plt.close('all')
    print('  wrote', fname)


def method_legend_strip(fname, ncol=5):
    "Standalone method legend (nine color patches) shared by panels c-f, wrapped over two rows."
    paper_panel(1)                                                # set the shared paper font sizes
    handles = [Patch(facecolor=f4.PALETTE[m], label=m) for m in f4.ALL]
    fig = plt.figure(figsize=(180 / 25.4, 18 / 25.4))            # tight bbox crops to the legend
    fig.legend(handles=handles, loc='center', ncol=ncol, frameon=False,
               handlelength=1.2, handleheight=1.1, handletextpad=0.4, columnspacing=1.4)
    save_svg(fname)
    plt.close('all')
    print('  wrote', fname)


def build(nk_cap, prefix):
    """Generate the full six-panel Fig2 set at one num_kin cap, writing `{prefix}{a..f}_*.svg`.

    nk_cap = su.NK_MAIN (the main figure, the reliably annotated <= 10 test subset) or None (the
    supplement, computed on every test site). The prefix keeps the two versions in separate files.
    """
    set_sns()
    pools, _ = su.load_pool()
    window_panel(pools, 'ST', FIG / f'{prefix}a_window_ST.svg', nk_cap)
    window_panel(pools, 'Tyr', FIG / f'{prefix}b_window_Tyr.svg', nk_cap)

    pairs = load_pairs_full(nk_cap)
    ostats = overall_stats(pairs)
    overall_panel(ostats, 'ST', 'ST', FIG / f'{prefix}c_overall_ST.svg')
    overall_panel(ostats, 'Tyr', 'TK', FIG / f'{prefix}d_overall_TK.svg')

    group_panel(group_stats(pairs, macro=False), 'micro recall@10', FIG / f'{prefix}e_pergroup_micro.svg')
    group_panel(group_stats(pairs, macro=True), 'macro recall@10', FIG / f'{prefix}f_pergroup_macro.svg')
    method_legend_strip(FIG / f'{prefix}_method_legend.svg')


def main():
    build(su.NK_MAIN, 'fig2')            # main figure: the reliably annotated num_kin <= 10 test subset


if __name__ == '__main__':
    main()
