"""kd_fig7_panels · Build the paper's kinase-domain prediction panels: main Fig. 7 + Supplementary Fig. 2.

Each panel is sized with `kplot.utils.paper_panel(frac, ratio)` to the fixed 180 x 170 mm page, 7 pt
fonts, 0.6 pt frame, SVG (place at 100%). Method colors follow the scoring palette (PSPA blue, CDDM
green, MLP orange); status colors match kd_08 (Known red / Predicted green / Unknown gray). Data comes
from the earlier kd_ scripts (kd_04b/04c grid scores, kd_06b LOO curve, kd_08 UMAP features, kd_09
validation), so the panels regenerate from data.

Paper panels (function -> panel):
  Fig. 7  b  panel_d      proximity cutoff vs nn_dist (LOO)
          c  panel_e      coverage UMAP (Known / Predicted / Unknown)
          d  panel_h      external-validation dumbbell (CDDM Pearson; PSPA AP@5 variant)
          e  panel_fg 'e' PRKACA (cow) example heatmaps
          f  panel_fg 'f' PRKCA (cow) example heatmaps
  Supp Fig. 2  f  panel_b('pspa')  model x feature dot plot (PSPA)
               g  panel_b('cddm')  model x feature dot plot (CDDM)
               h  train_example    PAK6 target vs predicted (run separately, not in main())
Fig. 7a and Supp Fig. 2e are hand-drawn schematics (not here); Supp Fig. 2 a-c (UMAPs) come from
motif_08 and Supp Fig. 2d (silhouette) from motif_07. `fig7supp_ap_dumbbell.svg` is an extra
CDDM+PSPA AP@5 validation view.

Inputs   out/kd_grid_scores.parquet, out/kd_dnn_scores.parquet (kd_04b/04c); kd_train_*_onehot (kd_03);
         raw/t5_kd.parquet + out/kd_pred_new_* (kd_07/08); raw/psp_ks_dataset_2408.csv (kd_09);
         out/kd_validation_{nonhuman,by_kinase}.csv (kd_09)
Outputs  fig/suppfig2{f,g}_model_feature_{pspa,cddm}.svg, fig/fig7b_proximity_cutoff_{pspa,cddm}.svg,
         fig/fig7c_umap_t5.svg, fig/fig7{e,f}_<gene>_heatmaps.svg, fig/fig7d_dumbbell_*.svg,
         fig/fig7supp_ap_dumbbell.svg

Run:  python nbs/kd_fig7_panels.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from paths import FIG, OUT, RAW

import kd_06b_confidence_threshold as k6
import kd_09_validate_nonhuman as val
import kd_util
from katlas.plot import plot_pssm_heatmaps
from katlas.utils import sty_color
from kplot.scatter import reduce_feature
from kplot.utils import paper_panel, save_svg, set_sns

#: method colors = scoring_04d palette (PSPA blue, CDDM green, MLP orange)
METHOD_COL = {'PSPA': '#4292c6', 'CDDM': '#41ab5d', 'MLP-attr': '#f16913'}
STATUS_COL = {'Known': '#d62728', 'Predicted': '#2ca02c', 'Unknown': '#c2c2c2'}   # kd_08
STTK_COL = {'S/T': sty_color['s'], 'TK': sty_color['y']}                           # S/T = pS/T blue, TK = pY orange
FEATS = ['onehot', 'onehot_pca', 'esm', 't5']
FEAT_COL = {'onehot': '#c0392b', 'onehot_pca': '#e0913c', 'esm': '#5b8fb0', 't5': '#7d3c98'}
SUPP_PANEL = {'pspa': 'f', 'cddm': 'g'}          # Supp Fig 2 letter for each model x feature target


# ---------------------------------------------- Supp Fig 2 f/g (panel_b): model x feature -----------------
def panel_b(target='pspa', metric='pearson', show_selected=True, feat_legend_loc='lower left', feat_legend_ncol=4):
    "Model x feature dot plot (overall flank Pearson, S/T); kNN x onehot boxed. show_selected labels that marker."
    g = pd.concat([pd.read_parquet(OUT / 'kd_grid_scores.parquet'),
                   pd.read_parquet(OUT / 'kd_dnn_scores.parquet')], ignore_index=True)
    g = g[(g.target == target) & (g.group != 'TK')]
    per = g.groupby(['model', 'feature', 'repeat'])[metric].median().reset_index()
    summ = per.groupby(['model', 'feature'])[metric].agg(point='mean', err='std').reset_index()
    order = summ.groupby('model').point.max().sort_values(ascending=False).index.tolist()
    xoff = dict(zip(FEATS, [-0.26, -0.09, 0.09, 0.26]))
    sel = summ[(summ.model == 'kNN') & (summ.feature == 'onehot')].iloc[0]
    mean_base = summ[summ.model == 'mean'].point.iloc[0]

    fig, ax = plt.subplots(figsize=paper_panel(1 / 2, ratio=1.75))
    if show_selected:                                    # dashed line marks the selected kNN x onehot level
        ax.axhline(sel.point, color='0.55', ls='--', lw=0.8, zorder=0)
    ax.axhline(mean_base, color='0.75', ls=':', lw=0.8, zorder=0)
    for i, m in enumerate(order):
        for _, r in summ[summ.model == m].iterrows():
            s = (m == 'kNN' and r.feature == 'onehot')
            ax.errorbar(i + xoff[r.feature], r.point, yerr=r.err, fmt='o', ms=5 if s else 3,
                        color=FEAT_COL[r.feature], ecolor=FEAT_COL[r.feature], elinewidth=0.6, capsize=1,
                        mec='#111' if s else 'none', mew=1.1 if s else 0, zorder=3)
    ax.set_xticks(range(len(order)), order, rotation=35, ha='right')
    ax.set_xlim(-0.6, len(order) - 0.4); ax.margins(y=0.04)
    ax.set_ylabel(f'overall flank Pearson ({target.upper()}, S/T)')
    feat_leg = ax.legend(handles=[Line2D([0], [0], marker='o', ls='', ms=4, color=FEAT_COL[f], label=f) for f in FEATS],
                         loc=feat_legend_loc, ncol=feat_legend_ncol, frameon=False, handletextpad=0.2, columnspacing=0.9)
    ax.add_artist(feat_leg)
    if show_selected:                                    # label the boxed kNN x onehot marker (Supp Fig 2f only)
        ax.legend(handles=[Line2D([0], [0], marker='o', ls='', ms=5, mfc=FEAT_COL['onehot'], mec='#111',
                                  mew=1.1, label='selected model-feature combination')],
                  loc='upper right', frameon=False, handletextpad=0.2)
    fig.tight_layout(pad=0.3)
    save_svg(FIG / f'suppfig2{SUPP_PANEL[target]}_model_feature_{target}.svg'); plt.close('all')


# ---------------------------------------------------- Fig 7 b (panel_d): proximity cutoff ----------------
def panel_d(target='pspa'):
    "Overall flank Pearson vs nn_dist (LOO kNN). The one conservative cutoff (PSPA's) is applied to all "
    "targets; this target's own, looser elbow is drawn faint for reference."
    thr = pd.read_parquet(OUT / 'kd_confidence_threshold.parquet').set_index('target')
    cut = float(thr.loc[target, 'threshold'])        # ONE shared cutoff (PSPA's 11.83) applied to every target
    b_high = float(thr.loc[target, 'b_high']); b_med = float(thr.loc[target, 'b_med'])
    own = float(thr.loc[target, 'own_elbow'])        # this target's own elbow (>= cut for cddm / mlp_attr)

    df, feat_col, target_col = kd_util.load_train(target, 'onehot')
    X, Y = df[feat_col].to_numpy(float), df[target_col].to_numpy(float)
    ids = df.iloc[:, 0].to_numpy()
    grp = kd_util.kinase_taxonomy(pd.Index(ids)).set_index('kinase')['group'].reindex(ids).to_numpy()
    P, nnd = k6.loo_retrieval(X, Y)
    pear = kd_util.pssm_scores(Y, P, target_col)[2]
    st = grp != 'TK'
    cv = k6.binned_curve(nnd[st], pear[st])

    fig, ax = plt.subplots(figsize=paper_panel(1 / 2, ratio=2.1))
    for x0, x1, col in [(0, b_high, '#2ca02c'), (b_high, b_med, '#e0b000'), (b_med, cut, '#e67e22')]:
        ax.axvspan(x0, x1, color=col, alpha=0.11, lw=0, zorder=0)
    ax.scatter(nnd[~st], pear[~st], s=5, c=STTK_COL['TK'], alpha=0.45, lw=0, label='TK', zorder=2)
    ax.scatter(nnd[st], pear[st], s=5, c=STTK_COL['S/T'], alpha=0.45, lw=0, label='S/T', zorder=2)
    ax.plot(cv.hi, cv.s, '-o', color='#22303a', lw=1.1, ms=2.6, zorder=4)
    ax.axvline(cut, color='#111', ls='--', lw=1, zorder=5, label=f'applied cutoff = {cut:.2f}')
    if abs(own - cut) > 0.05:                         # cddm / mlp_attr: own elbow is looser than the applied cutoff
        ax.axvline(own, color='0.55', ls=':', lw=1, zorder=5, label=f'{target.upper()} own elbow = {own:.2f}')
    ax.set_xlabel('nn_dist to nearest labeled kinase'); ax.set_ylabel('overall flank Pearson')
    ax.set_xlim(0, None); ax.margins(x=0.01)
    ax.legend(loc='lower left', frameon=False, handletextpad=0.2, markerscale=1.6, fontsize=5.5)
    fig.tight_layout(pad=0.3)
    save_svg(FIG / f'fig7b_proximity_cutoff_{target}.svg'); plt.close('all')


# ------------------------------------------------------- Fig 7 c (panel_e): coverage UMAP -------------------
def panel_e():
    "ProtT5 UMAP of all active domains, colored Known / Predicted / Unknown (seed 123, same as kd_08)."
    feat = pd.read_parquet(RAW / 't5_kd.parquet')
    known, predicted = set(), set()
    for label in ['pspa', 'cddm', 'mlp_attr']:
        known |= set(pd.read_parquet(OUT / f'kd_train_{label}_t5.parquet').index)
        p = OUT / f'kd_pred_new_{label}.parquet'
        if p.exists():
            pr = pd.read_parquet(p); predicted |= set(pr.index[pr.predictable.to_numpy()])
    embed = reduce_feature(feat, method='umap', complexity=30, min_dist=0.6)
    status = pd.Series('Unknown', index=embed.index)
    status.loc[embed.index.isin(predicted)] = 'Predicted'
    status.loc[embed.index.isin(known)] = 'Known'
    counts = status.value_counts()

    fig, ax = plt.subplots(figsize=paper_panel(1 / 2, ratio=1.5))
    xc, yc = embed.columns[:2]
    # Unknown = faint gray backdrop; Known vs Predicted distinguished by shape (circle vs triangle)
    layers = [('Unknown',   dict(s=2, marker='o', c='#d0d0d0', zorder=1)),
              ('Predicted', dict(s=2, marker='^', c=STATUS_COL['Predicted'], alpha=0.7, zorder=2)),
              ('Known',     dict(s=2, marker='o', c=STATUS_COL['Known'], alpha=0.7, zorder=3))]
    for k, kw in layers:
        sub = embed[status.values == k]
        ax.scatter(sub[xc], sub[yc], lw=0, label=f'{k} (n={counts[k]:,})', **kw)
    ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2'); ax.set_xticks([]); ax.set_yticks([])
    h, l = ax.get_legend_handles_labels()
    o = [l.index(f'{k} (n={counts[k]:,})') for k in ['Known', 'Predicted', 'Unknown']]
    ax.legend([h[i] for i in o], [l[i] for i in o], loc='lower left', frameon=False,
              handletextpad=0.2, borderpad=0.1, labelspacing=0.25, markerscale=4)
    fig.tight_layout(pad=0.3)
    save_svg(FIG / 'fig7c_umap_t5.svg'); plt.close('all')


# ------------------------------------------------------ Fig 7 e/f (panel_fg): example heatmaps --------------
def panel_fg(kin, gene, panel, psp_all, label):
    "PSP-observed vs predicted PSPA/CDDM/MLP-attr for one kinase, via katlas.plot.plot_pssm_heatmaps."
    pssms = {'PSP (observed)': psp_all.loc[kin],             # flat Series -> auto-recovered by the plotter
             'PSPA': val.load_pred('pspa').loc[kin],
             'CDDM': val.load_pred('cddm').loc[kin],
             'MLP-attr': val.load_pred('mlp_attr').loc[kin]}
    fig, _ = plot_pssm_heatmaps(pssms, fill_ps_from_pt=['PSPA'],   # PSPA flank pS duplicates pT
                                figsize=paper_panel(2 / 3, ratio=2.2), title=label)
    save_svg(FIG / f'fig7{panel}_{gene.lower()}_heatmaps.svg'); plt.close('all')


# --------------------------------------- Supp Fig 2 h (train_example): PAK6 target vs predicted ------
def train_example(kin_id, gene, targets=('pspa', 'cddm')):
    """Measured target PSSM vs the kNN model's prediction (k=5, one-hot), for one labeled kinase.
    The kinase is predicted from the OTHER labeled kinases (retrieved from the reference set, exactly as
    the deployed model predicts a new kinase). One figure per method (PSPA, CDDM), each a
    Target | Predicted heatmap pair (viridis). Illustrates the features -> model -> PSSM pipeline.
    Paper Supplementary Fig. 2h is this run for PAK6; call train_example('<PAK6 id>', 'PAK6') to build it.
    Output keeps its semantic name kd_train_example_<gene>_<method>.svg (a general helper, not PAK6-only)."""
    for t in targets:
        df, feat_col, target_col = kd_util.load_train(t, 'onehot')
        ids = df.iloc[:, 0].astype(str).to_numpy()
        if kin_id not in set(ids):
            print(f'  {kin_id} not in {t} labeled set - skip', flush=True); continue
        X, Y = df[feat_col].to_numpy(float), df[target_col].to_numpy(float)
        P, _ = k6.loo_retrieval(X, Y)                             # k=5 distance-weighted kNN, query from the others
        i = int(np.where(ids == kin_id)[0][0])
        pssms = {'Target': pd.Series(Y[i], index=target_col), 'Predicted': pd.Series(P[i], index=target_col)}
        fill = ['Target', 'Predicted'] if t == 'pspa' else None   # PSPA flank pS duplicates pT
        w, h = paper_panel(1 / 3, ratio=1.15)                      # set paper styling, then narrow the width only
        plot_pssm_heatmaps(pssms, fill_ps_from_pt=fill,
                           figsize=(w * 0.82, h), title=f'{gene} ({t.upper()})')
        save_svg(FIG / f'kd_train_example_{gene.lower()}_{t}.svg'); plt.close('all')


# ------------------------------------------- Fig 7 d (panel_h) + fig7supp: external dumbbell --------
def _dumbbell_styled(uni, per, method, metric, mlab, fname):
    "One-method validation dumbbell; color = S/T vs TK group, marker fill = genuine / mixed / identical-ortholog status."
    grp = uni[('info', 'group')]
    genes = list(uni[(method, metric)].dropna().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=paper_panel(1 / 2, ratio=1.55))
    for i, gg in enumerate(genes):
        v = uni.loc[gg, (method, metric)]
        c = STTK_COL['TK'] if grp.loc[gg] == 'TK' else STTK_COL['S/T']
        sv = per.loc[per.gene == gg, (method, metric)].dropna().to_numpy()
        err = sv.std(ddof=1) if len(sv) > 1 else 0
        nn = per.loc[per.gene == gg, ('info', 'nn_dist')].to_numpy()
        ax.errorbar(i, v, yerr=err, fmt='none', ecolor=c, elinewidth=0.6, capsize=1.2, zorder=3)
        if (nn > 0).all():
            fs, mfc, alt = 'full', c, c
        elif (nn == 0).all():
            fs, mfc, alt = 'full', 'white', 'white'
        else:
            fs, mfc, alt = 'left', c, 'white'                # mixed: half filled
        ax.plot(i, v, marker='o', ms=4.5, fillstyle=fs, mfc=mfc, mfcalt=alt, mec=c, mew=0.8, ls='', zorder=4)
    gray = '#777777'                                          # fill-style legend keys, color-neutral
    ax.legend(handles=[
        Line2D([0], [0], marker='o', ls='', ms=5, mfc=STTK_COL['S/T'], mec=STTK_COL['S/T'], label='S/T kinase'),
        Line2D([0], [0], marker='o', ls='', ms=5, mfc=STTK_COL['TK'], mec=STTK_COL['TK'], label='TK'),
        Line2D([0], [0], marker='o', ls='', ms=5, fillstyle='full', mfc=gray, mec=gray, label='genuine prediction (nn_dist > 0)'),
        Line2D([0], [0], marker='o', ls='', ms=5, fillstyle='left', mfc=gray, mfcalt='white', mec=gray, mew=0.8, label='mixed (some species identical)'),
        Line2D([0], [0], marker='o', ls='', ms=5, fillstyle='full', mfc='white', mec=gray, mew=0.8, label='identical ortholog (nn_dist = 0)')],
        loc='lower left', frameon=False, handletextpad=0.2, fontsize=5.0, labelspacing=0.3)
    nspec = {gg: int(per.loc[per.gene == gg, (method, metric)].notna().sum()) for gg in genes}
    ax.set_xticks(range(len(genes)), [f'{gg} (n={nspec[gg]})' for gg in genes], rotation=90, fontsize=5)
    ax.set_xlim(-0.7, len(genes) - 0.3); ax.set_ylim(0, 1.02); ax.set_ylabel(mlab)
    ax.tick_params(bottom=True, length=1.5, pad=1)
    fig.tight_layout(pad=0.3)
    save_svg(FIG / f'{fname}.svg'); plt.close('all')


def panel_h():
    "External validation dumbbells. Main: CDDM overall Pearson (same-method vs PSP); PSPA: AP@5 (cross-method)."
    uni = pd.read_csv(OUT / 'kd_validation_by_kinase.csv', header=[0, 1], index_col=0)
    per = pd.read_csv(OUT / 'kd_validation_nonhuman.csv', header=[0, 1], index_col=0)
    per = per.assign(gene=per[('info', 'kinase')].values)

    # main Fig 7d: CDDM vs PSP-observed is same-method -> overall flank Pearson
    _dumbbell_styled(uni, per, 'CDDM', 'pearson', 'overall flank Pearson (vs PSP)', 'fig7d_dumbbell_cddm_pearson')
    # Fig 7d PSPA version: PSPA vs PSP-observed is cross-method -> AP@5 (rank metric), not Pearson
    _dumbbell_styled(uni, per, 'PSPA', 'ap', 'AP@5 (vs PSP)', 'fig7d_dumbbell_pspa_ap')

    # supp: CDDM + PSPA AP@5 side by side
    genes = list(uni[('CDDM', 'ap')].dropna().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=paper_panel(1 / 2, ratio=1.55))
    for meth, dx in zip(['CDDM', 'PSPA'], [-0.14, 0.14]):
        for i, gg in enumerate(genes):
            v = uni.loc[gg, (meth, 'ap')]
            if not np.isfinite(v):
                continue
            sv = per.loc[per.gene == gg, (meth, 'ap')].dropna().to_numpy()
            err = sv.std(ddof=1) if len(sv) > 1 else 0
            ax.errorbar(i + dx, v, yerr=err, fmt='o', ms=3.5, color=METHOD_COL[meth],
                        ecolor=METHOD_COL[meth], elinewidth=0.6, capsize=1.2, zorder=3)
    ax.legend(handles=[Line2D([0], [0], marker='o', ls='', ms=4, color=METHOD_COL[m], label=m) for m in ['CDDM', 'PSPA']],
              loc='lower left', frameon=False, handletextpad=0.2)
    nspec = {gg: int(per.loc[per.gene == gg, ('CDDM', 'ap')].notna().sum()) for gg in genes}
    ax.set_xticks(range(len(genes)), [f'{gg} (n={nspec[gg]})' for gg in genes], rotation=90, fontsize=5)
    ax.set_xlim(-0.7, len(genes) - 0.3); ax.set_ylim(0, 1.02); ax.set_ylabel('AP@5 (vs PSP)')
    ax.tick_params(bottom=True, length=1.5, pad=1)
    fig.tight_layout(pad=0.3)
    save_svg(FIG / 'fig7supp_ap_dumbbell.svg'); plt.close('all')


def main():
    set_sns()
    panel_b('pspa')                                                     # Supp Fig 2f: selected label + legend lower-left
    panel_b('cddm', show_selected=False, feat_legend_loc='upper right', feat_legend_ncol=2)  # Supp Fig 2g: no label, legend right
    print('  suppfig2f/g model_feature (pspa + cddm) done', flush=True)
    panel_d('pspa')
    panel_d('cddm')
    print('  fig7b proximity_cutoff (pspa + cddm) done', flush=True)
    panel_e()
    print('  fig7c umap done', flush=True)
    psp_all, _ = val.nonhuman_pssms()
    for panel, kin, gene, label in [('e', 'P00517', 'PRKACA', 'PRKACA (cow)'),   # cow orthologs, nn_dist>0
                                    ('f', 'P04409', 'PRKCA', 'PRKCA (cow)')]:
        panel_fg(kin, gene, panel, psp_all, label)
    print('  fig7e/f heatmaps done', flush=True)
    panel_h()
    print('  fig7d (cddm + pspa) + fig7supp done', flush=True)
    print('wrote fig7* + suppfig2* panels to', FIG)


if __name__ == '__main__':
    main()
