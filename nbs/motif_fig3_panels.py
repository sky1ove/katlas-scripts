"""motif_fig3 · Figure-3 panel generator: CDDM / MLP-attr / PSPA systematic comparison (ST-kinase agreement).

Parallel to kd_fig7_panels.py. Panels sized via kplot.utils.paper_panel to the fixed 180x170 mm page,
7 pt fonts, 0.6 pt frame, SVG (place at 100%). Method colors follow the scoring palette (PSPA blue,
CDDM green, MLP-attr orange).

The figure compares the three ways to write a kinase's motif — CDDM (from observed substrates), MLP-attr
(the distilled attribution), and PSPA (the array) — and shows they agree, most on the ST groups.

Paper Fig. 3 panels (function -> panel):
  a   panel_a: S/T-ratio joint scatter (kplot.scatter.jointscatter), CDDM vs PSPA per Ser/Thr kinase.
  b   panel_c: AP@5 vs PSPA by kinase group, CDDM freq vs MLP-attr only (the CDDM log-odds series is
      dropped to keep the comparison to CDDM, MLP-attr, PSPA). Groups ordered by overall agreement
      high->low, with TK forced last so the ST groups (high agreement) contrast with TK (low). Error
      bars are bootstrap 95% CIs over the group's kinases.
  c   representatives + heatmap_panel: per-group representative-kinase heatmaps (PSPA / CDDM / MLP-attr)
      in the kd-7e/f style: the highest-agreement kinase of each group, three matrices side by side.
Auxiliary (not a panel in the current Fig. 3): panel_b, CDDM 0y vs TK/non-TK acceptor (fig3_acceptor.svg).

Inputs   out/compare_vs_pspa_bygroup.csv (motif_16, per-kinase AP vs PSPA); out/cddm_acceptor_roc.parquet
         (motif_01f); kdata: cddm, pspa, pspa_scale, pspa_enrich, kinase_info; out/mlp_attr_pssm_full.parquet
Outputs  fig/fig3a_st_ratio.svg, fig/fig3b_ap.svg, fig/fig3c_<gene>_heatmaps.svg (+ auxiliary fig/fig3_acceptor.svg)

Run:  python nbs/motif_fig3_panels.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paths import FIG, OUT

import kdata
from katlas.plot import plot_pssm_heatmaps
from katlas.pssm import recover_pssm
from katlas.utils import group_color
from kplot.scatter import jointscatter
from kplot.utils import paper_panel, save_svg, set_sns

METHOD_COL = {'CDDM': '#41ab5d', 'MLP-attr': '#f16913', 'PSPA': '#4292c6'}
BY = OUT / 'compare_vs_pspa_bygroup.csv'
ATTR = OUT / 'mlp_attr_pssm_full.parquet'


def _group_of():
    return kdata.load('kinase_info').drop_duplicates('kinase').set_index('kinase')['group']


def _ap_table():
    "Per-kinase AP@5 vs PSPA for CDDM freq and MLP-attr, on the kinases that have both, with group."
    d = pd.read_csv(BY)
    d = d[d.method.isin(['CDDM freq', 'CDDM MLP-attr'])].copy()
    both = set(d[d.method == 'CDDM freq'].kinase) & set(d[d.method == 'CDDM MLP-attr'].kinase)
    d = d[d.kinase.isin(both)]
    d['group'] = d.kinase.map(_group_of())
    d['method'] = d.method.map({'CDDM freq': 'CDDM', 'CDDM MLP-attr': 'MLP-attr'})
    return d.dropna(subset=['group'])


def _order(d):
    "Kinase groups by overall (both-method) mean AP@5 descending, TK forced last (ST vs TK contrast)."
    gm = d.groupby('group').ap.mean().sort_values(ascending=False)
    non_tk = [g for g in gm.index if g != 'TK']
    return non_tk + (['TK'] if 'TK' in gm.index else [])


def _boot_ci(vals, n=2000, seed=0):
    "Bootstrap 95% CI of the mean over a group's per-kinase values."
    v = np.asarray(vals, float)
    if len(v) < 2:
        return v.mean(), v.mean()
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


# ---------------------------------------------------------- Fig 3a (panel_a): S/T-ratio agreement ----
def _st_ratio_data():
    "log2(0s/0t) acceptor preference per Ser/Thr kinase from CDDM and PSPA, with Modi group (no pseudo)."
    cddm, pspa = kdata.load('cddm'), kdata.load('pspa_scale')
    cddm = cddm[pspa.columns]
    pspa_st = pspa[pspa['0y'] == 0]                          # Ser/Thr kinases only (drop any Tyr-centre)
    st = lambda m: np.log2(m['0s'] / m['0t'])
    r = pd.concat([st(pspa_st), st(cddm)], axis=1)
    r.columns = ['pspa', 'cddm']
    info = kdata.load('kinase_info')
    info = info[info.pseudo == '0']
    r['group'] = r.index.map(info.set_index('kinase')['group'])
    return r.replace([np.inf, -np.inf], np.nan).dropna()


def panel_a(frac=0.42, ratio=1.3):
    "S/T-ratio agreement (CDDM vs PSPA): grouped joint scatter (kplot.scatter.jointscatter). "
    "`frac` is the panel's share of the 180 mm width (a bit over 1/3 to give the legend room)."
    r = _st_ratio_data()
    fig, _ = jointscatter(r, 'pspa', 'cddm', hue='group', palette=group_color,
                          figsize=paper_panel(frac, ratio=ratio), lim=[-3.7, 5.7], bins=25,
                          xlabel='Log2(S/T) from PSPA', ylabel='Log2(S/T) from CDDM',
                          spearman_fontsize=7)
    save_svg(FIG / 'fig3a_st_ratio.svg')
    plt.close(fig)


# ------------------------------------------ auxiliary (not in Fig 3): CDDM 0y acceptor vs TK/non-TK ----
def panel_b(frac=1 / 3, ratio=1.3):
    "CDDM central Y fraction (0y) by TK / non-TK class, dual-specificity (_TYR) kinases marked. No cut line."
    d = pd.read_parquet(OUT / 'cddm_acceptor_roc.parquet').set_index('kinase')
    TK_COL, OTHER_COL, DUAL_COL = '#c0392b', '#3a6ea5', '#e08e2a'
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=paper_panel(frac, ratio=ratio))
    for cls, color, label in [(0, OTHER_COL, 'non-TK'), (1, TK_COL, 'TK')]:
        s = d[(d.is_tk == cls) & (~d.dual_specificity)]
        ax.scatter(cls + rng.normal(0, 0.06, len(s)), s.cddm_0y, s=8, color=color, alpha=.55,
                   edgecolor='none', label=label)
    dd = d[d.dual_specificity]                                # all sit in ST-side (non-TK) Modi groups
    ax.scatter(dd.is_tk + rng.normal(0, 0.06, len(dd)), dd.cddm_0y, s=16, marker='D', color=DUAL_COL,
               edgecolor='k', lw=.3, zorder=4, label='dual-spec (_TYR)')
    for k in ('WEE1', 'LIMK1'):                               # the two dual-spec with high 0y
        if k in dd.index:
            ax.annotate(k, (dd.loc[k, 'is_tk'], dd.loc[k, 'cddm_0y']), xytext=(6, 0),
                        textcoords='offset points', fontsize=6, va='center', color='0.2')
    for k, row in d[(d.is_tk == 1) & (d.cddm_0y < 0.6)].iterrows():   # the lone TK with low 0y (MATK)
        ax.annotate(k, (row.is_tk, row.cddm_0y), xytext=(6, 5), textcoords='offset points',
                    fontsize=6, va='bottom', color=TK_COL)
    ax.set_xticks([0, 1], ['non-TK', 'TK'])
    ax.set_xlim(-0.5, 1.9); ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel('CDDM central Y fraction (0y)', labelpad=2)
    ax.tick_params(length=2, pad=1.5)
    ax.legend(fontsize=5, frameon=False, loc='lower right', handletextpad=.3, labelspacing=.3)
    save_svg(FIG / 'fig3_acceptor.svg')
    plt.close(fig)


# ------------------------------------------------------------------- Fig 3b (panel_c): AP@5 by group ----
def panel_c(frac=2 / 3, ratio=3.2, style='bar'):
    "AP@5 vs PSPA by group, CDDM vs MLP-attr, per-kinase dots overlaid. "
    "style='bar' (bar + bootstrap-CI + dots) or 'violin' (violin + dots). Returns (d, order)."
    d = _ap_table()
    order = _order(d)
    n = d.kinase.nunique()
    methods = ['CDDM', 'MLP-attr']
    x = np.arange(len(order))
    w = 0.40
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=paper_panel(frac, ratio=ratio))
    for i, meth in enumerate(methods):
        col = METHOD_COL[meth]
        pos = x + (i - 0.5) * w
        vals = [d[(d.group == g) & (d.method == meth)].ap.values for g in order]
        if style == 'violin':
            ok = [(p, v) for p, v in zip(pos, vals) if len(v) > 1]
            parts = ax.violinplot([v for _, v in ok], positions=[p for p, _ in ok], widths=w * .95,
                                  showextrema=False)
            for b in parts['bodies']:
                b.set_facecolor(col); b.set_alpha(.30); b.set_edgecolor('none')
            ax.plot([], [], color=col, lw=4, alpha=.5, label=meth)          # legend proxy
        else:                                                               # bar + bootstrap CI
            means, los, his = [], [], []
            for v in vals:
                m = float(np.mean(v)) if len(v) else np.nan
                lo, hi = _boot_ci(v)
                means.append(m); los.append(m - lo); his.append(hi - m)
            ax.bar(pos, means, w, color=col, alpha=.5, label=meth, zorder=1,
                   yerr=[los, his], error_kw=dict(lw=.6, capsize=1.5, capthick=.6, zorder=2))
        for p, v in zip(pos, vals):                                         # per-kinase dots (both styles)
            ax.scatter(p + rng.normal(0, w * .11, len(v)), v, s=2.5, color=col, alpha=.55,
                       edgecolor='none', zorder=3)
    ax.set_xticks(x, order, rotation=35, ha='right')
    ax.set_ylabel('AP@5 vs PSPA', labelpad=2)
    ax.tick_params(length=2, pad=1.5)                                   # short ticks, tight tick-to-label gap
    ax.set_title(f'CDDM & MLP-attr agreement with PSPA by group (n={n})', fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.margins(x=0.01)
    save_svg(FIG / ('fig3b_ap_violin.svg' if style == 'violin' else 'fig3b_ap.svg'))
    plt.close('all')
    return d, order


# ----------------------------------------------------------- Fig 3c: representative-kinase heatmaps ----
def _load_mats():
    drop = lambda d: d[~d.index.str.contains('_TYR')]
    return (drop(kdata.load('pspa_enrich')), drop(kdata.load('pspa')),
            kdata.load('cddm'), pd.read_parquet(ATTR))


def _pspa_display(kin, enr, pspa_orig):
    "PSPA matrix for the heatmap: enrich flank (signed, shows depletion) but the position-0 acceptor is "
    "taken from the original pspa (positive, no depletion), kept only in the lowercase s/t/y rows so it "
    "renders red/white and max-rescaled exactly like CDDM's position 0."
    m = recover_pssm(enr.loc[kin])
    m0 = recover_pssm(pspa_orig.loc[kin])
    m[0] = 0.0
    for r in ('s', 't', 'y'):
        if r in m.index and r in m0.index:
            m.loc[r, 0] = m0.loc[r, 0]
    return m


def heatmap_panel(kin, gene, enr, pspa_orig, cddm, attr, label):
    "Three matrices (PSPA / CDDM / MLP-attr) for one kinase via katlas.plot.plot_pssm_heatmaps."
    pssms = {'PSPA': _pspa_display(kin, enr, pspa_orig), 'CDDM': cddm.loc[kin], 'MLP-attr': attr.loc[kin]}
    # all three on the diverging map (CDDM has no negatives -> white->red only); flank = enrichment, but
    # the position-0 acceptor of PSPA & CDDM is max-rescaled to the flank (larger of S/T -> full scale).
    # 1/3-width, a touch flattened; smaller tick font keeps all 23 AA labels legible.
    fig, _ = plot_pssm_heatmaps(pssms, fill_ps_from_pt=['PSPA'], signed=True,
                                scale_acceptor=['PSPA', 'CDDM'], drop_zero=True, title=label,
                                figsize=paper_panel(1 / 3, ratio=1.4), tick_fontsize=4.5)
    save_svg(FIG / f'fig3c_{gene.lower()}_heatmaps.svg')
    plt.close(fig)


def representatives(d, order):
    "Highest-agreement kinase (mean of CDDM & MLP-attr AP) per group, in the panel order."
    per = d.groupby(['group', 'kinase']).ap.mean().reset_index()
    top = per.sort_values('ap', ascending=False).groupby('group').first()
    return [(top.loc[g, 'kinase'], g, round(float(top.loc[g, 'ap']), 2)) for g in order if g in top.index]


def main():
    set_sns()
    panel_a()
    panel_b()
    d, order = panel_c(style='violin')   # Fig 3b: violin + dots -> fig3b_ap_violin.svg
    print('Fig 3b (panel_c): group order =', order)

    reps = representatives(d, order)
    print('representative kinase per group (highest AP@5):')
    for kin, g, ap in reps:
        print(f'  {g:>8}: {kin}  (AP@5 {ap})')

    enr, pspa_orig, cddm, attr = _load_mats()
    for kin, g, ap in reps:
        if all(kin in mat.index for mat in (enr, pspa_orig, cddm, attr)):
            heatmap_panel(kin, kin, enr, pspa_orig, cddm, attr, f'{kin} ({g})')
        else:
            print(f'  skip {kin} ({g}): missing a matrix')
    print('wrote fig3a_st_ratio.svg, fig3b_ap_violin.svg, fig3_acceptor.svg (aux) + per-group fig3c_<kinase>_heatmaps.svg')


if __name__ == '__main__':
    main()
