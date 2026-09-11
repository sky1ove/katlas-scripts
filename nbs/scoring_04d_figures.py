"""scoring_04d · The paper figures.

Assembles the benchmark result from the persisted pairs of scoring_04a / 04b / 04c (methods) and
scoring_02a / 02b / 02c (window sweeps). Nine rows: two baselines plus the seven scored methods.

  Random  a uniformly random rank - the floor
  Dummy   rank by training frequency - the floor a model must beat to have learned anything
          about sequence rather than about which kinases are common

**Reported over the headline RepeatedStratifiedGroupKFold (K=5 × R=3).** The method pairs stamp
`seed = repeat`; within a repeat the 5 folds cover every site once, and the 3 repeats pool to the
point estimate. The **error bar is a kinase cluster-bootstrap 95% CI** (the biological unit is the
kinase), the same honest interval the paper panels use (`scoring_fig2_panels`), computed here by the
shared `overall_stats` / `group_stats` / `macro_by_window_ci` helpers. Significance is a per-kinase
paired Wilcoxon (printed) with a bootstrap 95% CI on the median per-kinase Δ, which does not rely on
the repeat count.
(The window sweeps 02a/b/c keep the legacy 3-seed scheme — a supporting analysis.)

**MAIN RESULT = the num_kin ≤ 10 test subset.** Sites hit by more than 10 kinases have unreliable
annotations (a whole subfamily can phosphorylate a site labelled for one member), so the metric is
reported where the ground truth is trustworthy. The candidate pool K is unchanged — only the test
set is filtered.

Three metrics only: micro recall@10, macro recall@10, AUCDF. Micro
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
#: point from the curve - macro_by_window_ci asserts instead.
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


# ---------- kinase cluster-bootstrap 95% CI (biological replicate = the kinase) ----------
# Canonical CI computation for the whole benchmark: scoring_fig2_panels imports these (as f4.*) for the
# paper panels, so both figure sets use ONE honest error bar, a cluster bootstrap over kinases (the
# independent unit), never the SD-across-repeats that understates uncertainty for near-deterministic methods.

def _xerr(d, key):
    "Asymmetric [below, above] error lengths from *_lo / *_hi CI columns (NaN CI -> 0 length)."
    m = d[key].to_numpy(float)
    e = np.vstack([m - d[f'{key}_lo'].to_numpy(float), d[f'{key}_hi'].to_numpy(float) - m])
    return np.clip(np.nan_to_num(e, nan=0.0), 0, None)


def overall_stats(pairs):
    "Per (branch, method): mean + kinase cluster-bootstrap 95% CI for AUCDF, micro and macro recall@10."
    perk = su.summarize(pairs, ['method', 'branch', 'kinase'])
    rows = []
    for br in ['ST', 'Tyr']:
        d = pairs[pairs.branch == br]
        agg = su.summarize(d, ['method', 'branch']).set_index('method')       # pooled (micro) points
        for m in ALL:
            if m not in agg.index:
                continue
            ci = su.boot_micro_ci(d[d.method == m])                           # kinase cluster bootstrap
            kv = perk[(perk.method == m) & (perk.branch == br)]['top10'].to_numpy()
            mlo, mhi = su.boot_macro_ci(kv)
            rows.append({'branch': br, 'method': m,
                         'AUCDF': agg.loc[m, 'AUCDF'], 'AUCDF_lo': ci['AUCDF'][0], 'AUCDF_hi': ci['AUCDF'][1],
                         'micro': agg.loc[m, 'top10'], 'micro_lo': ci['top10'][0], 'micro_hi': ci['top10'][1],
                         'macro': float(kv.mean()), 'macro_lo': mlo, 'macro_hi': mhi})
    return pd.DataFrame(rows)


def group_stats(pairs, macro):
    "Per (kinase group, method): mean + kinase cluster-bootstrap 95% CI for micro (pooled) or macro recall@10."
    rows = []
    if macro:
        perk = su.summarize(pairs, ['method', 'kinase_group', 'kinase'])
        for (m, g), sub in perk.groupby(['method', 'kinase_group'], observed=True):
            kv = sub['top10'].to_numpy()
            lo, hi = su.boot_macro_ci(kv)
            rows.append({'kinase_group': g, 'method': m, 'mean': float(kv.mean()), 'lo': lo, 'hi': hi})
    else:
        agg = su.summarize(pairs, ['method', 'kinase_group']).set_index(['method', 'kinase_group'])
        for (m, g), sub in pairs.groupby(['method', 'kinase_group'], observed=True):
            lo, hi = su.boot_micro_ci(sub).get('top10', (np.nan, np.nan))
            rows.append({'kinase_group': g, 'method': m, 'mean': agg.loc[(m, g), 'top10'], 'lo': lo, 'hi': hi})
    return pd.DataFrame(rows)


def macro_by_window_ci(fname, split, wmap, metric='top10', nk_cap=None):
    "Per (window, branch): macro recall@10 (mean over kinases) with a bootstrap 95% CI over kinases."
    p = pd.read_parquet(su.RES / fname)
    p = p[p.split == split]
    p = (p if nk_cap is None else p[p.num_kin <= nk_cap]).copy()
    p['w'] = p.window.map(wmap) if wmap else p.window.astype(int)
    unmapped = sorted(p.loc[p.w.isna(), 'window'].unique())
    assert not unmapped, f'{fname}: window labels missing from the map: {unmapped}'
    perk = su.summarize(p, ['w', 'branch', 'kinase'])
    rows = []
    for (w, br), g in perk.groupby(['w', 'branch'], observed=True):
        kv = g[metric].to_numpy()
        lo, hi = su.boot_macro_ci(kv, seed=int(w))
        rows.append({'w': w, 'branch': br, 'mean': float(kv.mean()), 'lo': lo, 'hi': hi})
    return pd.DataFrame(rows).set_index(['w', 'branch'])


def overall_grid(stats, metrics, fname, suptitle):
    "Method on y, ST/TK rows, one column per metric; error bar = kinase cluster-bootstrap 95% CI."
    fig, axes = plt.subplots(2, len(metrics), figsize=(4.8 * len(metrics), 7.0),
                             squeeze=False, sharey=True)
    y = np.arange(len(ALL))[::-1]                                # ALL[0] (Random) at the top
    colors = [PALETTE[m] for m in ALL]
    for ri, (brk, brl) in enumerate([('ST', 'ST'), ('Tyr', 'TK')]):
        d = stats[stats.branch == brk].set_index('method').reindex(ALL)
        for ci, (mk, ml) in enumerate(metrics):
            ax = axes[ri][ci]
            ax.barh(y, d[mk].to_numpy(float), color=colors, height=0.72)
            ax.errorbar(d[mk].to_numpy(float), y, xerr=_xerr(d, mk), fmt='none',
                        ecolor='0.4', elinewidth=1, capsize=2)                 # kinase-bootstrap 95% CI
            ax.set_title(f'{brl}: {ml}', fontsize=11)
            ax.set_xlabel('')
            ax.set_ylabel('')
            if mk == 'AUCDF':
                ax.set_xlim(0.45, None)
            ax.grid(axis='x', alpha=0.3)
        axes[ri][0].set_yticks(y)
        axes[ri][0].set_yticklabels(ALL)
    sns.despine(fig)
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_svg(fname)
    plt.close('all')
    print('  wrote', fname)


def draw_window_fig(pools, split, fname, metric='top10', mlabel='recall@10'):
    "Macro recall@10 vs flank window; shaded band = kinase cluster-bootstrap 95% CI over kinases."
    curves = {k: macro_by_window_ci(v[0], split, v[1], metric, nk_cap=su.NK_MAIN) for k, v in WIN.items()}
    xticks = sorted({int(x) for c in curves.values() for x in c.index.get_level_values('w')})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, br in zip(axes, ['ST', 'Tyr']):
        for name, (_, _, col, mk) in WIN.items():
            d = curves[name].xs(br, level='branch').sort_index()
            x = d.index.to_numpy()
            ax.fill_between(x, d['lo'], d['hi'], color=col, alpha=0.15, linewidth=0)   # 95% CI over kinases
            ax.plot(x, d['mean'], mk, color=col, ms=6, lw=1.5, label=name)
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
    fig.suptitle(f'macro {mlabel} vs window, {split.upper()} set (kinase-bootstrap 95% CI band)', fontsize=11)
    plt.tight_layout()
    save_svg(fname)
    plt.close('all')
    print('  wrote', fname)


def per_group_figs(pairs):
    "Per-kinase-group recall@10, nine dodged bars per group; error bar = kinase cluster-bootstrap 95% CI."
    gorder = [g for g in GORDER if g in pairs.kinase_group.unique()]

    def group_bar(stats, fname, title, ylabel):
        nm = len(ALL); w = 0.8 / nm
        S = stats.set_index(['kinase_group', 'method'])
        fig, ax = plt.subplots(figsize=(18, 5.5))
        for j, m in enumerate(ALL):
            xs, ys, elo, ehi = [], [], [], []
            for gi, g in enumerate(gorder):
                if (g, m) in S.index:
                    r = S.loc[(g, m)]
                    xs.append(gi + (j - (nm - 1) / 2) * w)
                    ys.append(r['mean'])
                    elo.append(0.0 if np.isnan(r['lo']) else r['mean'] - r['lo'])
                    ehi.append(0.0 if np.isnan(r['hi']) else r['hi'] - r['mean'])
            ax.bar(xs, ys, width=w, color=PALETTE[m], linewidth=0, label=m)
            if xs:
                ax.errorbar(xs, ys, yerr=np.vstack([elo, ehi]), fmt='none',
                            ecolor='0.4', elinewidth=0.8, capsize=0)             # kinase-bootstrap 95% CI
        ax.set_xticks(range(len(gorder)))
        ax.set_xticklabels(gorder, fontsize=11)
        ax.set_xlim(-0.5, len(gorder) - 0.5)
        ax.set_ylim(0, None)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(ncol=len(ALL), fontsize=8, loc='lower center',
                  bbox_to_anchor=(0.5, 1.02), frameon=False)
        sns.despine(ax=ax)
        plt.tight_layout()
        save_svg(fname)
        plt.close('all')

    base = 'per kinase group, test (mean, kinase-bootstrap 95% CI)'
    group_bar(group_stats(pairs, macro=False), FIG / 'pergroup_micro.svg', f'micro recall@10 {base}', 'micro recall@10')
    group_bar(group_stats(pairs, macro=True), FIG / 'pergroup_macro.svg', f'macro recall@10 {base}', 'macro recall@10')
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

    ostats = overall_stats(pairs)
    print(ostats.set_index(['branch', 'method'])[['micro', 'macro', 'AUCDF']].round(3).to_string())

    main_title = 'test (mean, kinase-bootstrap 95% CI)'
    overall_grid(ostats, [('micro', 'micro recall@10'), ('macro', 'macro recall@10')],
                 FIG / 'overall_micro_macro.svg', f'Overall: {main_title}')
    overall_grid(ostats, [('AUCDF', 'AUCDF')], FIG / 'overall_aucdf.svg',
                 f'Overall AUCDF: {main_title}')

    if all((su.RES / v[0]).exists() for v in WIN.values()):
        for split_name in ['val', 'test']:
            draw_window_fig(pools, split_name, FIG / f'window_macro_{split_name}.svg')
    else:
        print('window figures skipped - run scoring_02a / scoring_02b / scoring_02c first')

    per_group_figs(pairs)


if __name__ == '__main__':
    main()
