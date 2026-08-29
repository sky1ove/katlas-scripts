"""scoring_04d · The paper figures.

Assembles the benchmark result from the persisted pairs of scoring_04a / 04b / 04c (methods) and
scoring_02a / 02b / 02c (window sweeps). Nine rows: two baselines plus the seven scored methods.

  Random  a uniformly random rank - the floor
  Dummy   rank by training frequency - the floor a model must beat to have learned anything
          about sequence rather than about which kinases are common

**Reported over the headline RepeatedStratifiedGroupKFold (K=5 × R=3).** The method pairs stamp
`seed = repeat`; within a repeat the 5 folds cover every site once (one full-CV estimate), so the
point estimate is the mean over repeats and the **error bar is the SD across the 3 repeats**
(reproducibility under re-shuffle), NOT a parametric 95% CI over three non-independent 20% draws.
Significance is a per-kinase paired Wilcoxon (printed) with a bootstrap 95% CI on the median per-kinase
Δ (resampling kinases — the biological unit), which does not rely on the repeat count.
(The window sweeps 02a/b/c keep the legacy 3-seed scheme — a supporting analysis.)

**MAIN RESULT = the num_kin ≤ 10 test subset.** Sites hit by more than 10 kinases have unreliable
annotations (a whole subfamily can phosphorylate a site labelled for one member), so the metric is
reported where the ground truth is trustworthy. The candidate pool K is unchanged — only the test
set is filtered.

Three metrics only: micro recall@10, macro recall@10, AUCDF (recall@5 as a supplement). Micro
weights every test pair equally; macro averages per kinase first, so rare kinases count as much as
common ones.

Figures: overall micro+macro, overall AUCDF, macro@10 vs window (validation and test), and the
same broken down per kinase group.

Inputs   out/scoring_pairs/{pspa,cddm,cddm_seq}_pairs.parquet, window_{mlp,cddm,pspa}_pairs.parquet,
         out/scoring_split.parquet, out/scoring_pool.parquet
Outputs  fig/*.svg

Run:  python nbs/scoring_04d_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import scoring_util as su
import seaborn as sns
from matplotlib import pyplot as plt
from paths import FIG

from kplot.bar import plot_group_bar
from kplot.utils import save_svg, set_sns

#: source method name -> display name
NAME = {'PSPA: phospho': 'PSPA', 'PSPA: phospho + pct': 'PSPA pct',
        'CDDM (w5)': 'CDDM', 'CDDM (w5) pct': 'CDDM pct',
        'CDDM-seq Linear': 'CDDM-seq Linear', 'CDDM-seq MLP': 'CDDM-seq MLP',
        'CDDM-seq MLP + PSPA': 'CDDM-seq MLP + PSPA'}
ORDER = list(NAME.values())
BASE = ['Random', 'Dummy']
ALL = BASE + ORDER
PALETTE = {'Random': '#d9d9d9', 'Dummy': '#8a8a8a',
           'PSPA': '#9ecae1', 'PSPA pct': '#4292c6',
           'CDDM': '#a1d99b', 'CDDM pct': '#41ab5d',
           'CDDM-seq Linear': '#fdae6b', 'CDDM-seq MLP': '#f16913',
           'CDDM-seq MLP + PSPA': '#a63603'}
GORDER = ['AGC', 'CAMK', 'CK1', 'CMGC', 'STE', 'TKL', 'NEK', 'Atypical', 'Other', 'TK']

#: PSPA's window labels are categorical (see scoring_02c's WINS); map them onto the numeric axis
#: the other two sweeps use. An unmapped label would be dropped by the groupby, silently losing a
#: point from the curve - macro_by_window asserts instead.
PW = {'±0': 0, '±1': 1, '±2': 2, '±3': 3, '±4': 4, 'full(-5..+4)': 5}
WIN = {'CDDM PSSM (generative)': ('window_cddm_pairs.parquet', None, '#238b45', '-o'),
       'CDDM-seq MLP (discriminative)': ('window_mlp_pairs.parquet', None, '#f16913', '-s'),
       'PSPA (experimental)': ('window_pspa_pairs.parquet', PW, '#4292c6', '-^')}


def load_pairs():
    "The seven scored methods, restricted to the main num_kin subset and renamed for display."
    raw = pd.concat([pd.read_parquet(su.RES / f)
                     for f in ['pspa_pairs.parquet', 'cddm_pairs.parquet', 'cddm_seq_pairs.parquet']],
                    ignore_index=True)
    missing = sorted(set(NAME) - set(raw.method))
    assert not missing, f'missing methods: {missing}'
    pairs = raw[raw.method.isin(NAME)].copy()
    pairs['method'] = pairs.method.map(NAME)
    pairs = pairs[pairs.num_kin <= su.NK_MAIN].copy()
    print(f'pairs: {len(pairs):,} (num_kin <= {su.NK_MAIN})')
    return pairs


def baseline_pairs(pairs, split, pools, kind):
    """Random / Dummy as pseudo-methods, so every metric is computed by the same code path.

    Borrows the PSPA rows' test-pair layout (site / kinase / group / num_kin / n_pool) and just
    replaces the rank: uniform for Random, training-frequency order for Dummy.
    """
    out = []
    ref = pairs[pairs.method == 'PSPA']
    for (br, seed), g in ref.groupby(['branch', 'seed']):
        pool = pools[br]
        K = len(pool)
        rng = np.random.default_rng(seed)
        if kind == 'Random':
            r = rng.integers(1, K + 1, len(g))
        else:
            # Dummy floor = rank by how common each kinase is, from TRAIN only (never test). seed = repeat
            # here, and the legacy test_<seed> column marks a valid leak-free 80% train partition; its exact
            # fold layout is irrelevant for a marginal-popularity floor, but it must not count the held-out
            # sites (a global train+test frequency would let the floor peek at test labels).
            tr = split[~split[f'test_{seed}']]
            fr = tr[tr.kinase_protein.isin(pool)].kinase_protein.value_counts().reindex(pool).fillna(0)
            frank = {k: i + 1 for i, k in enumerate(fr.sort_values(ascending=False).index)}
            r = g.kinase.map(frank).to_numpy()
        gg = g.copy()
        gg['rank'] = r.astype(np.int32)
        gg['ap'] = np.nan
        gg['method'] = kind
        out.append(gg)
    return pd.concat(out, ignore_index=True)


def with_macro(pairs, keys, metric):
    "Micro (over pairs) plus macro (per-kinase mean) for one metric."
    o = su.summarize(pairs, keys)
    mac = (su.summarize(pairs, keys + ['kinase']).groupby(keys)[metric]
           .mean().rename('macro').reset_index())
    o = o.merge(mac, on=keys)
    o['micro'] = o[metric]
    return o


def overall_grid(dfov, metrics, fname, suptitle):
    "Method on y, ST/TK rows, one column per metric; error bar = SD across the 3 CV repeats."
    fig, axes = plt.subplots(2, len(metrics), figsize=(4.8 * len(metrics), 7.0),
                             squeeze=False, sharey=True)
    for ri, (brk, brl) in enumerate([('ST', 'ST'), ('Tyr', 'TK')]):
        d = dfov[dfov.branch == brk]
        for ci, (mk, ml) in enumerate(metrics):
            ax = axes[ri][ci]
            # saturation=1.0: seaborn desaturates to 0.75 by default, which dulls PALETTE.
            # errorbar='sd': spread across the 3 repeats (each a full 5-fold CV) — reproducibility
            # under re-shuffle, not a parametric CI over non-independent draws.
            sns.barplot(data=d, y='method', x=mk, order=ALL, hue='method', hue_order=ALL,
                        palette=PALETTE, saturation=1.0, legend=False, errorbar='sd',
                        capsize=.3, err_kws={'lw': 1, 'color': '0.4'}, ax=ax, orient='h')
            ax.set_title(f'{brl} — {ml}', fontsize=11)
            ax.set_xlabel('')
            ax.set_ylabel('')
            if mk == 'AUCDF':
                ax.set_xlim(0.45, None)
            ax.grid(axis='x', alpha=0.3)
    sns.despine(fig)
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_svg(fname)
    plt.close('all')
    print('  wrote', fname)


def macro_by_window(fname, split, wmap, metric):
    p = pd.read_parquet(su.RES / fname)
    p = p[(p.num_kin <= su.NK_MAIN) & (p.split == split)].copy()
    p['w'] = p.window.map(wmap) if wmap else p.window.astype(int)
    unmapped = sorted(p.loc[p.w.isna(), 'window'].unique())
    assert not unmapped, f'{fname}: window labels missing from the map: {unmapped}'
    return (su.summarize(p, ['w', 'branch', 'seed', 'kinase'])
            .groupby(['w', 'branch', 'seed'])[metric].mean()
            .groupby(['w', 'branch']).agg(['mean', 'std']))


def draw_window_fig(pools, split, fname, metric='top10', mlabel='recall@10'):
    curves = {k: macro_by_window(v[0], split, v[1], metric) for k, v in WIN.items()}
    xticks = sorted({int(x) for c in curves.values() for x in c.index.get_level_values('w')})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, br in zip(axes, ['ST', 'Tyr']):
        for name, (_, _, col, mk) in WIN.items():
            d = curves[name].xs(br, level='branch').sort_index()
            ax.errorbar(d.index, d['mean'], yerr=d['std'].fillna(0), fmt=mk, color=col, ms=6,
                        lw=1.5, elinewidth=1.3, capsize=3, capthick=1.3, label=name)
        ax.set_xticks(xticks)
        ax.tick_params(axis='x', labelsize=9)
        chance = 10 / len(pools[br])
        ax.axhline(chance, ls='--', color='0.6', lw=1)
        ax.text(0.98, chance, 'random', fontsize=7, color='0.6', ha='right', va='bottom',
                transform=ax.get_yaxis_transform())
        ax.set_xlabel('window half-width w (±w; PSPA "full" = −5..+4 at w=5)')
        ax.set_ylabel(f'macro {mlabel}')
        ax.set_title(br)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle(f'macro {mlabel} vs window — {split.upper()} set', fontsize=11)
    plt.tight_layout()
    save_svg(fname)
    plt.close('all')
    print('  wrote', fname)


def per_group_figs(pairs):
    gorder = [g for g in GORDER if g in pairs.kinase_group.unique()]
    micro = su.summarize(pairs, ['method', 'seed', 'kinase_group'])
    macro = (su.summarize(pairs, ['method', 'seed', 'kinase_group', 'kinase'])
             .groupby(['method', 'seed', 'kinase_group'])[['top5', 'top10']].mean().reset_index())

    def group_bar(long, valcol, fname, title, ylabel):
        wide = long.pivot_table(index=['kinase_group', 'seed'], columns='method',
                                values=valcol).reset_index()
        ax = plot_group_bar(wide, value_cols=ALL, group='kinase_group', order=gorder,
                            figsize=(18, 5.5), rotation=0, fontsize=11, title=title,
                            palette=PALETTE, hue_order=ALL, saturation=1.0, errorbar='sd')
        ax.set_ylabel(ylabel, fontsize=12)
        save_svg(fname)
        plt.close('all')

    for m, tag, lbl in [('top10', '', 'recall@10'), ('top5', '_r5', 'recall@5')]:
        base = 'per kinase group — test (mean ± SD, 5-fold × 3-repeat CV)'
        group_bar(micro.rename(columns={m: 'v'}), 'v', FIG / f'pergroup_micro{tag}.svg',
                  f'micro {lbl} {base}', f'micro {lbl}')
        group_bar(macro.rename(columns={m: 'v'}), 'v', FIG / f'pergroup_macro{tag}.svg',
                  f'macro {lbl} {base}', f'macro {lbl}')
    print('  wrote fig/pergroup_*.svg')


def paired_wilcoxon(pairs, a='CDDM-seq MLP + PSPA',
                    baselines=('CDDM', 'CDDM pct', 'PSPA', 'PSPA pct'), metric='top10'):
    """Per-kinase paired Wilcoxon signed-rank: does the top discriminative method `a` beat EVERY
    prespecified generative baseline (display names; incl. raw CDDM, the strongest Ser/Thr generative),
    per branch, with Holm–Bonferroni adjustment across the baselines within a branch. Independent of the
    repeat count, so it is the significance statement behind the ranking — and the 'beats every
    generative baseline' claim needs the weakest (largest-p) comparison to survive."""
    perk = su.summarize(pairs, ['method', 'branch', 'kinase'])       # per-kinase, pooled over repeats
    print('two-sided Wilcoxon signed-rank (paired, per-kinase, num_kin <= %d; Holm-Bonferroni per branch):'
          % su.NK_MAIN)
    for br in ['ST', 'Tyr']:
        wide = perk[perk.branch == br].pivot_table(index='kinase', columns='method', values=metric)
        recs = []
        for i, b in enumerate(baselines):
            if a in wide and b in wide:
                x = wide[[a, b]].dropna()
                recs.append((b, len(x), su.wilcoxon_report(x[a].to_numpy(), x[b].to_numpy(), seed=i)))
        ps = [r[2]['p'] for r in recs]; order = sorted(range(len(ps)), key=lambda i: ps[i]); m = len(ps)
        adj = [0.0] * m; run = 0.0                                  # Holm-Bonferroni step-down
        for rank, i in enumerate(order):
            run = max(run, (m - rank) * ps[i]); adj[i] = min(run, 1.0)
        for (b, n, rep), pa in zip(recs, adj):
            cid = rep['ci']
            ci = '' if any(np.isnan(cid)) else f' [95% CI {cid[0]:+.3f}, {cid[1]:+.3f}]'
            print(f'  {br}: {a} vs {b:9s} — W={rep["W"]:.0f}, n={n}, r={rep["r"]:+.2f}, '
                  f'median Δ{metric}={rep["median_delta"]:+.3f}{ci}, p={rep["p"]:.1e}, p_holm={pa:.1e}')


def main():
    set_sns()
    split = su.load_split()
    pools, _ = su.load_pool()

    pairs = load_pairs()
    pairs = pd.concat([pairs,
                       baseline_pairs(pairs, split, pools, 'Random'),
                       baseline_pairs(pairs, split, pools, 'Dummy')], ignore_index=True)
    paired_wilcoxon(pairs)

    keys = ['method', 'seed', 'branch']
    ov = with_macro(pairs, keys, 'top10')
    ov5 = with_macro(pairs, keys, 'top5')
    print(ov.groupby(['branch', 'method'])[['micro', 'macro', 'AUCDF']]
          .mean().round(3).to_string())

    main_title = 'test (mean ± SD, 5-fold × 3-repeat CV)'
    overall_grid(ov, [('micro', 'micro recall@10'), ('macro', 'macro recall@10')],
                 FIG / 'overall_micro_macro.svg', f'Overall — {main_title}')
    overall_grid(ov5, [('micro', 'micro recall@5'), ('macro', 'macro recall@5')],
                 FIG / 'overall_micro_macro_r5.svg',
                 f'Overall recall@5 [supplementary] — {main_title}')
    overall_grid(ov, [('AUCDF', 'AUCDF')], FIG / 'overall_aucdf.svg',
                 'Overall AUCDF — test (supplement)')

    if all((su.RES / v[0]).exists() for v in WIN.values()):
        for split_name in ['val', 'test']:
            draw_window_fig(pools, split_name, FIG / f'window_macro_{split_name}.svg')
            draw_window_fig(pools, split_name, FIG / f'window_macro_{split_name}_r5.svg',
                            'top5', 'recall@5')
    else:
        print('window figures skipped — run scoring_02a / scoring_02b / scoring_02c first')

    per_group_figs(pairs)


if __name__ == '__main__':
    main()
