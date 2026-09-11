"""pathway_08 · Quantitative figure: predicted substrates recover kinase function.

The kinome-wide validation panel for the pathway analysis, on the chosen pipeline: site-centric
target selection, local ORA vs the phosphoproteome background, raw-p specificity ranking (enrichment
minus the kinome mean). The claim is quantitative because per-kinase top-N pathway bars stay generic
even under the best method - the recovery lives in the full ranking, which AUROC captures.

  a  per-kinase pathway-recovery AUROC for CDDM (one dot per kinase, Ser/Thr vs Tyr), against the
     permutation null (each kinase scored against a random kinase's profile). Above the null band =
     predicted substrates recover the kinase's own Reactome pathways.
  b  recovery signal (mean AUROC - null) per scoring method, split S/T vs TK. Two versions are
     written: CDDM vs PSPA, and CDDM vs PSPA vs MLP - pick one.

Inputs   out/pathway_{cddm,pspa,mlp}_sc_info.parquet (pathway_01b / pathway_06), kdata: kinase_info;
         Data.reactome_pathway  (via pathway_07.universe / ora_matrix)
Outputs  fig/pathway_recovery_cddm_pspa.svg, fig/pathway_recovery_cddm_pspa_mlp.svg,
         fig/pathway_recovery_by_group{,_violin}.svg (3 methods x kinase group, per-group null),
         fig/pathway_recovery_overall_{shared,all}.svg (overall dots+violin; shared vs all-available),
         out/pathway_cddm_perkinase_auroc.parquet

Run:  python nbs/pathway_08_figure.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from paths import FIG, OUT
from sklearn.metrics import roc_auc_score

import kdata
from kplot.utils import paper_panel, save_svg
from pathway_07_local_ora import ora_matrix, universe

MIN_REF = 10
MIN_GROUP = 5           # a kinase group needs this many shared kinases for a stable per-group null
N_PERM = 100
SEED = 0
METHOD_TAG = {'PSPA': 'pspa_sc', 'CDDM': 'cddm_sc', 'MLP': 'mlp_sc'}   # display order
METHOD_COL = {'PSPA': '#4292c6', 'CDDM': '#41ab5d', 'MLP': '#f16913'}
STY = {'S/T kinases': '#4c72b0', 'Tyr kinases': '#dd8452'}


def per_kinase(spec, ann, kin_uni):
    "AUROC per kept kinase (own annotated pathways vs rest), plus the y-vectors for the null."
    idx = spec.index; y, keep = {}, []
    for k in spec.columns:
        refs = ann.get(kin_uni.get(k), set()) & set(idx)
        if len(refs) >= MIN_REF:
            y[k] = idx.isin(refs).astype(int); keep.append(k)
    au = pd.Series({k: roc_auc_score(y[k], spec[k].values) for k in keep})
    return au, y


def _derange(items, rng):
    "A permutation of `items` with no fixed point: each kinase is scored against a DIFFERENT kinase's profile."
    items = list(items)
    if len(items) < 2:
        return items
    while True:
        perm = list(items); rng.shuffle(perm)
        if all(a != b for a, b in zip(items, perm)):
            return perm


def split_signal(spec, au, y, group):
    "Per split (S/T, TK): mean AUROC, permutation null (within split), signal."
    out = {}
    for lab, sel in [('S/T', [k for k in au.index if group.get(k) != 'TK']),
                     ('TK', [k for k in au.index if group.get(k) == 'TK'])]:
        rng = np.random.default_rng(SEED); nulls = []
        for _ in range(N_PERM):
            perm = _derange(sel, rng)
            nulls.append(np.mean([roc_auc_score(y[k], spec[kp].values) for k, kp in zip(sel, perm)]))
        m = au[sel].mean(); nu = float(np.mean(nulls))
        out[lab] = {'auroc': m, 'null': nu, 'signal': m - nu, 'n': len(sel)}
    return out


def _perm_null(sel, spec_m, y_m):
    "Mean AUROC over `sel` when each kinase is scored against a random kinase's profile."
    rng = np.random.default_rng(SEED); nl = []
    for _ in range(N_PERM):
        perm = _derange(sel, rng)
        nl.append(np.mean([roc_auc_score(y_m[k], spec_m[kp].values) for k, kp in zip(sel, perm)]))
    return float(np.mean(nl))


def _violin_third(df, ylabel, methods, n, out_svg, title=None, null_by_method=None):
    "Single 1/3-width dots+violin panel (kplot); null_by_method draws a per-method null line."
    from kplot.bar import plot_group_violin
    fig, ax = plt.subplots(figsize=paper_panel(1 / 3, ratio=1.0))
    plot_group_violin(df, value='value', group='method', order=methods, palette=METHOD_COL,
                      ax=ax, dot_size=1.6, ylabel=ylabel)
    if null_by_method is not None:                            # raw-value panel: null line per method
        for xi, m in enumerate(methods):
            ax.plot([xi - 0.42, xi + 0.42], [null_by_method[m]] * 2, color='0.35', lw=1.0, ls='--')
    else:                                                     # null-corrected panel: null is 0
        ax.axhline(0, color='0.5', lw=0.7, ls='--')

    # per method: a median tick (central tendency) and the % of kinases above the null (win rate,
    # outlier-proof, since the mean can be pulled by a few high-AUROC kinases)
    ymin, ymax = df.value.min(), df.value.max()
    ax.set_ylim(ymin - 0.04 * (ymax - ymin), ymax + 0.10 * (ymax - ymin))
    for xi, m in enumerate(methods):
        v = df[df.method == m].value.values
        ref = null_by_method[m] if null_by_method is not None else 0.0
        ax.plot([xi - 0.22, xi + 0.22], [np.median(v)] * 2, color='0.1', lw=1.3, zorder=6)
        ax.text(xi, ax.get_ylim()[1], f'{100 * np.mean(v > ref):.0f}%',
                ha='center', va='top', fontsize=6, color='0.25')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([f'{m}\n(n={n[m]})' for m in methods])
    if title:
        ax.set_title(title, loc='left', fontweight='bold')
    ax.tick_params(length=2, pad=1.5)
    fig.tight_layout()
    save_svg(out_svg)
    plt.close('all')
    print('  wrote', out_svg)


def build_overall_auroc(spec, au, yd, out_svg):
    "Overall: raw per-kinase AUROC (all available), with each method's permutation null drawn as a line."
    methods = list(au.keys())
    rows, n, nu = [], {}, {}
    for m in methods:
        sel = list(au[m].index); n[m] = len(sel); nu[m] = _perm_null(sel, spec[m], yd[m])
        rows += [{'method': m, 'value': au[m][k]} for k in sel]
    _violin_third(pd.DataFrame(rows), 'Pathway-recovery AUROC', methods, n, out_svg,
                  title='Overall', null_by_method=nu)


def build_subset_recovery(spec, au, yd, group, is_tk, title, out_svg):
    "One kinase class (S/T or TK): per-kinase recovery = AUROC minus that class's per-method null."
    methods = list(au.keys())
    rows, n = [], {}
    for m in methods:
        sel = [k for k in au[m].index if (group.get(k) == 'TK') == is_tk]; n[m] = len(sel)
        null = _perm_null(sel, spec[m], yd[m])
        rows += [{'method': m, 'value': au[m][k] - null} for k in sel]
    _violin_third(pd.DataFrame(rows), 'Recovery (AUROC−null)', methods, n, out_svg, title=title)


def build_group_violin(spec, au, yd, group, out_svg):
    "Per-kinase recovery (AUROC - the group's null) as grouped violins + dots (kplot), split by method. "
    "All available kinases per method (n differs per method, so no n on the axis). 2/3 width, ratio 3."
    from collections import Counter

    from kplot.bar import plot_group_violin
    from matplotlib.lines import Line2D
    methods = list(au.keys())                                 # PSPA, CDDM, MLP
    ccnt = Counter(group.get(k) for k in au['CDDM'].index)    # groups sized by CDDM's coverage
    groups = [g for g in ccnt if ccnt[g] >= MIN_GROUP]

    rows = []
    for g in groups:
        for m in methods:
            sel = [k for k in au[m].index if group.get(k) == g]
            if len(sel) < 2:
                continue
            rng = np.random.default_rng(SEED); nl = []
            for _ in range(N_PERM):
                perm = _derange(sel, rng)
                nl.append(np.mean([roc_auc_score(yd[m][k], spec[m][kp].values) for k, kp in zip(sel, perm)]))
            null = float(np.mean(nl))
            rows += [{'group': g, 'method': m, 'recovery': au[m][k] - null} for k in sel]
    df = pd.DataFrame(rows)
    order = sorted(groups, key=lambda g: -(df[(df.group == g) & (df.method == 'CDDM')].recovery.mean()))

    fig, ax = plt.subplots(figsize=paper_panel(2 / 3, ratio=3))
    plot_group_violin(df, value='recovery', group='group', hue='method', order=order,
                      hue_order=methods, palette=METHOD_COL, ax=ax, dot_size=1.3, legend=False,
                      ylabel='Recovery (AUROC−group null)')
    ax.axhline(0, color='0.5', lw=0.7, ls='--')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)                                 # group name only, no n
    handles = [Line2D([0], [0], marker='o', color=METHOD_COL[m], linestyle='none', markersize=4,
                      label=m) for m in methods]              # dot legend, not bar-style squares
    ax.legend(handles=handles, frameon=False, loc='upper right', ncol=3,
              handletextpad=0.2, columnspacing=0.9)
    ax.tick_params(length=2, pad=1.5)
    fig.tight_layout()
    save_svg(out_svg)
    plt.close('all')
    print('  wrote', out_svg)


def build_figure(methods, spec, sig, au_cddm, group, out_svg):
    paper_panel(1)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(180 / 25.4, 62 / 25.4),
                                   gridspec_kw={'width_ratios': [1.3, 1]})
    # a: CDDM per-kinase AUROC by class, with the per-class permutation null
    cd = au_cddm.rename('auroc').to_frame()
    cd['class'] = ['Tyr kinases' if group.get(k) == 'TK' else 'S/T kinases' for k in cd.index]
    order = ['S/T kinases', 'Tyr kinases']
    sns.stripplot(data=cd, x='class', y='auroc', order=order, hue='class', palette=STY,
                  size=2.3, jitter=0.28, alpha=0.6, legend=False, ax=axa)
    sns.boxplot(data=cd, x='class', y='auroc', order=order, width=0.5, showfliers=False,
                boxprops=dict(facecolor='none', edgecolor='0.3'), whiskerprops=dict(color='0.3'),
                capprops=dict(color='0.3'), medianprops=dict(color='0.1'), ax=axa)
    for xi, sp in zip((0, 1), ('S/T', 'TK')):                 # per-class null marker
        nu = sig['CDDM'][sp]['null']
        axa.plot([xi - 0.3, xi + 0.3], [nu, nu], color='0.4', lw=1.0, ls='--')
    axa.text(1.34, sig['CDDM']['TK']['null'], 'null', color='0.4', va='center', ha='left', fontsize=6)
    axa.set_ylabel('Pathway-recovery AUROC', labelpad=1); axa.set_xlabel('')
    axa.set_title('a  CDDM per-kinase recovery', loc='left', fontweight='bold')
    axa.tick_params(length=2, pad=1.5)
    for s in ('top', 'right'):
        axa.spines[s].set_visible(False)

    # b: recovery signal per method, grouped by split
    splits = ['S/T', 'TK']; xpos = np.arange(len(splits)); w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        vals = [sig[m][sp]['signal'] for sp in splits]
        axb.bar(xpos + i * w - 0.4 + w / 2, vals, w, color=METHOD_COL[m], label=m)
    axb.set_xticks(xpos); axb.set_xticklabels(splits)
    axb.set_ylabel('Recovery signal (AUROC − null)', labelpad=1)
    axb.set_title('b  scoring methods', loc='left', fontweight='bold')
    axb.legend(frameon=False, loc='upper right')
    axb.tick_params(length=2, pad=1.5)
    for s in ('top', 'right'):
        axb.spines[s].set_visible(False)

    fig.tight_layout()
    save_svg(out_svg)
    plt.close('all')
    print('  wrote', out_svg)


def main():
    bg, N, pw, pw_u, pw_K, names, ann = universe()
    info = kdata.load('kinase_info').drop_duplicates('kinase')
    kin_uni = info.set_index('kinase')['uniprot'].to_dict()
    group = info.set_index('kinase')['group'].to_dict()

    spec, sig, au, yd = {}, {}, {}, {}
    for m, tag in METHOD_TAG.items():
        M = ora_matrix(tag, bg, N, pw, pw_u, pw_K)
        spec[m] = M.sub(M.mean(axis=1), axis=0)
        au[m], yd[m] = per_kinase(spec[m], ann, kin_uni)
        sig[m] = split_signal(spec[m], au[m], yd[m], group)
        print(f'{m}: S/T signal {sig[m]["S/T"]["signal"]:+.3f} (n={sig[m]["S/T"]["n"]}) | '
              f'TK signal {sig[m]["TK"]["signal"]:+.3f} (n={sig[m]["TK"]["n"]})')

    au['CDDM'].rename('auroc').to_frame().assign(
        group=[group.get(k) for k in au['CDDM'].index]).to_parquet(
        OUT / 'pathway_cddm_perkinase_auroc.parquet')

    build_figure(['PSPA', 'CDDM'], spec, sig, au['CDDM'], group,
                 FIG / 'pathway_recovery_cddm_pspa.svg')
    build_figure(['PSPA', 'CDDM', 'MLP'], spec, sig, au['CDDM'], group,
                 FIG / 'pathway_recovery_cddm_pspa_mlp.svg')
    build_group_violin(spec, au, yd, group, FIG / 'pathway_recovery_by_group_violin.svg')
    build_overall_auroc(spec, au, yd, FIG / 'pathway_recovery_overall_auroc.svg')
    build_subset_recovery(spec, au, yd, group, False, 'S/T', FIG / 'pathway_recovery_st.svg')
    build_subset_recovery(spec, au, yd, group, True, 'TK', FIG / 'pathway_recovery_tk.svg')


if __name__ == '__main__':
    main()
