"""motif_01f · Does the CDDM central acceptor recover the tyrosine-kinase label?

CDDM position 0 records, from a kinase's OBSERVED substrate sites, the fraction whose phospho-
acceptor is S / T / Y (columns `0s` / `0t` / `0y`). A tyrosine kinase should phosphorylate tyrosine,
so `0y` alone ought to separate the TK kinases from the rest. This scores that directly: `0y` as a
single predictor of the binary is-TK label, by ROC/AUC (and average precision). It is a sanity check
that the observed-substrate signal recovers the known kinase-group taxonomy – no model, just the raw
acceptor fraction.

The label is `kinase_info.group == 'TK'`. `group` is the Modi classification (the one to use; the
traditional Manning labels are in `group_old`) – it is not called "Manning".

The comparison line is `pspa_scale`'s `0y` (the peptide array, per-position-normalised so `0y` is a
proper fraction) on the kinases it covers – does the independent assay draw the same boundary?

Both separate the classes near-perfectly; the interest is the handful of kinases in the overlap, and
those are exactly the dual-specificity kinases. We flag them the principled way – the `_TYR` suffix in
the PSPA index (`WEE1_TYR`, `LIMK1_TYR`, …), i.e. a kinase measured on the tyrosine array as well. All
15 sit in ST-side Modi groups (TKL / STE / Atypical / Other / NEK), none in TK – so dual specificity
overlaps the ST/non-TK side, not TK. The two that clear the Youden cut on CDDM `0y` (WEE1, which
phosphorylates CDK1 Tyr15; LIMK1) are in this `_TYR` set, so they read as tyrosine-directed by their
substrates even though their group is not TK – the acceptor identity is a property of the substrates,
not the label. The rest have modest `0y`.

The mirror case is the one TK sitting right at the cut – MATK (a genuine CSK-family tyrosine kinase,
NOT `_TYR` dual-specificity). Its `0y` is only ~0.42 because most of its annotated KS-dataset
substrate sites are S/T (0s ~0.45, 0t ~0.13), i.e. noisy/indirect substrate calls, not real dual
specificity – a reminder that CDDM's acceptor composition is only as clean as the site annotations.

Inputs   kdata: cddm, pspa_scale, kinase_info
Outputs  out/cddm_acceptor_roc.parquet – per kinase: group, is_tk, dual_specificity, cddm_0y, pspa_0y
         fig/cddm_0y_tk_roc.svg
         fig/cddm_0y_tk_separation.svg

Run:  python nbs/motif_01f_cddm_acceptor_roc.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paths import FIG, OUT
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import kdata
from kplot.utils import save_svg

TK_COL, OTHER_COL = '#c0392b', '#3a6ea5'      # tyrosine kinase vs the rest


def acceptor_frame():
    "Per-kinase is-TK label (Modi group) + dual-specificity flag + CDDM 0y and PSPA 0y."
    g = kdata.load('kinase_info').drop_duplicates('kinase').set_index('kinase')['group']  # Modi group
    cddm = kdata.load('cddm')
    pspa = kdata.load('pspa_scale')
    # dual-specificity kinases carry a `_TYR` PSPA row (measured on the tyrosine array too)
    dual = {k[:-4] for k in pspa.index if str(k).endswith('_TYR')}
    pspa = pspa[~pspa.index.str.contains('_TYR')]          # drop the dual-specificity _TYR duplicates

    df = pd.DataFrame({'cddm_0y': cddm['0y'], 'cddm_0s': cddm['0s'], 'cddm_0t': cddm['0t']})
    df['pspa_0y'] = pspa['0y'].reindex(df.index)
    df['group'] = df.index.map(g)
    df = df.dropna(subset=['group'])
    df['is_tk'] = (df.group == 'TK').astype(int)
    df['dual_specificity'] = df.index.isin(dual)           # _TYR-flagged, all in ST-side groups
    print(f'{len(df)} CDDM kinases with a group | TK {int(df.is_tk.sum())} · '
          f'non-TK {int((1 - df.is_tk).sum())} | PSPA-covered {int(df.pspa_0y.notna().sum())} | '
          f'dual-specificity (_TYR) present {int(df.dual_specificity.sum())}')
    return df


def score(df, col):
    "AUC + average precision for `col` as a predictor of is_tk, on the rows where `col` is present."
    d = df.dropna(subset=[col])
    auc = roc_auc_score(d.is_tk, d[col])
    ap = average_precision_score(d.is_tk, d[col])
    fpr, tpr, thr = roc_curve(d.is_tk, d[col])
    youden = thr[np.argmax(tpr - fpr)]                     # J-optimal cut, for the confusion report
    print(f'  {col:8} n={len(d):3d}  AUC={auc:.4f}  AP={ap:.4f}  Youden 0y*={youden:.3f}')
    return {'fpr': fpr, 'tpr': tpr, 'auc': auc, 'ap': ap, 'youden': youden, 'n': len(d)}


def plot_roc(sc):
    "ROC curves for CDDM 0y and (where available) PSPA 0y."
    plt.figure(figsize=(4.2, 4.2))
    for col, style, color in [('cddm_0y', '-', TK_COL), ('pspa_0y', '--', OTHER_COL)]:
        s = sc[col]
        plt.plot(s['fpr'], s['tpr'], style, color=color, lw=2,
                 label=f'{col}  AUC {s["auc"]:.4f} · AP {s["ap"]:.4f}  (n={s["n"]})')
    plt.plot([0, 1], [0, 1], ':', color='0.6', lw=1)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('0y predicts the tyrosine-kinase label', fontsize=11)
    plt.legend(loc='lower right', fontsize=7.5, frameon=False)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    save_svg(FIG / 'cddm_0y_tk_roc.svg')
    plt.close('all')
    print('  wrote', FIG / 'cddm_0y_tk_roc.svg')


def plot_separation(df, youden):
    "CDDM 0y by class, with the J-optimal cut and the dual-specificity (_TYR) kinases marked."
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    # plain (non-dual) points, split by the group label
    for cls, color, label in [(0, OTHER_COL, 'non-TK'), (1, TK_COL, 'TK (group)')]:
        d = df[(df.is_tk == cls) & (~df.dual_specificity)]
        x = cls + rng.normal(0, 0.06, len(d))
        ax.scatter(x, d.cddm_0y, s=16, color=color, alpha=0.55, edgecolor='none', label=label)
    # dual-specificity overlay (all sit in ST-side groups, i.e. non-TK) — the overlap zone
    dd = df[df.dual_specificity]
    xd = dd.is_tk + rng.normal(0, 0.06, len(dd))
    ax.scatter(xd, dd.cddm_0y, s=34, marker='D', color='#e08e2a', edgecolor='k', lw=0.4,
               zorder=4, label='dual-specificity (_TYR)')
    ax.axhline(youden, color='0.4', ls='--', lw=1)
    ax.text(-0.45, youden + 0.015, f'0y* = {youden:.2f}', va='bottom', ha='left', fontsize=8, color='0.4')

    # label the two that clear the cut inline; list the low-0y cluster compactly (their labels collide)
    hi = dd[dd.cddm_0y >= youden].sort_values('cddm_0y', ascending=False)
    lo = dd[dd.cddm_0y < youden].sort_values('cddm_0y', ascending=False)
    for k, r in hi.iterrows():
        ax.annotate(k, (r.is_tk, r.cddm_0y), xytext=(9, 0), textcoords='offset points',
                    fontsize=7.5, va='center', color='0.2')
    if len(lo):
        ax.text(-0.45, 0.30, 'low-0y dual-specificity:\n' + ', '.join(lo.index),
                fontsize=6.8, va='top', ha='left', color='#b5771f', linespacing=1.3)

    # the lone TK sitting at the cut: a genuine TK (not _TYR dual-spec) whose annotated substrates are
    # S/T-heavy — an annotation-quality artefact, labelled so it is not misread as an error
    for k, r in df[(df.is_tk == 1) & (df.cddm_0y < 0.6)].iterrows():
        ax.annotate(k, (r.is_tk, r.cddm_0y), xytext=(8, 7), textcoords='offset points',
                    fontsize=7.5, va='bottom', ha='left', color=TK_COL)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['non-TK', 'TK'])
    ax.set_xlim(-0.5, 2.0)
    ax.set_ylabel('CDDM central Y fraction (0y)')
    ax.set_title('Observed-substrate acceptor vs the Modi group label', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.6), fontsize=8, frameon=False)
    plt.tight_layout()
    save_svg(FIG / 'cddm_0y_tk_separation.svg')
    plt.close('all')
    print('  wrote', FIG / 'cddm_0y_tk_separation.svg')


def main():
    df = acceptor_frame()

    print('== 0y as a predictor of is_TK ==')
    sc = {c: score(df, c) for c in ['cddm_0y', 'pspa_0y']}

    # dual-specificity kinases: none are group TK — they sit in ST-side Modi groups
    dd = df[df.dual_specificity]
    print(f'\ndual-specificity (_TYR) kinases in CDDM: {len(dd)} | in TK group: {int(dd.is_tk.sum())} '
          f'| Modi groups: {dict(dd.group.value_counts())}')
    print('  ', ', '.join(f'{k}[{r.group}]={r.cddm_0y:.2f}'
                          for k, r in dd.sort_values('cddm_0y', ascending=False).iterrows()))

    # the TK kinases with the weakest observed-Y signal — genuine TKs whose annotated substrates are
    # S/T-heavy (dataset noise), NOT dual-specificity: they are not in the _TYR set
    tk_low = df[df.is_tk == 1].nsmallest(3, 'cddm_0y')
    print('\nlowest-0y TK (S/T-heavy substrate annotations, not _TYR dual-spec):',
          ', '.join(f'{k}=0y{r.cddm_0y:.2f}/0s{r.cddm_0s:.2f}/0t{r.cddm_0t:.2f}'
                    f'{"·DUAL" if r.dual_specificity else ""}'
                    for k, r in tk_low.iterrows()))

    yj = sc['cddm_0y']['youden']
    fp = df[(df.is_tk == 0) & (df.cddm_0y >= yj)].sort_values('cddm_0y', ascending=False)
    fn = df[(df.is_tk == 1) & (df.cddm_0y < yj)].sort_values('cddm_0y')
    print(f'\nat CDDM 0y* = {yj:.3f}: {len(fp)} non-TK above the cut, {len(fn)} TK below it')
    if len(fp):
        print('  non-TK above cut:', ', '.join(
            f'{k}[{r.group}{"·dual" if r.dual_specificity else ""}]={r.cddm_0y:.2f}'
            for k, r in fp.iterrows()))
    if len(fn):
        print('  TK below cut    :', ', '.join(f'{k}={r.cddm_0y:.2f}' for k, r in fn.iterrows()))

    out = OUT / 'cddm_acceptor_roc.parquet'
    df.reset_index(names='kinase').to_parquet(out, index=False)
    print('\nwrote', out, df.shape)

    plot_roc(sc)
    plot_separation(df, yj)


if __name__ == '__main__':
    main()
