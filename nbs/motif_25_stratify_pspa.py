"""motif_25_stratify_pspa · Does k-means stratification of CDDM recover agreement with PSPA?

For every kinase, compare the **overall CDDM** and each of its **k-means sub-motifs** (motif_01c)
against PSPA by two metrics over the ±5 flank: overall flank Pearson (whole-flank magnitude) and AP@5
(does the motif recover PSPA's 5 strongest flank cells). The question: which kinases have a sub-cluster
that agrees with PSPA *better than the whole CDDM does*, i.e. stratification isolates a PSPA-matching
sub-motif (this happens for the CDDM-vs-PSPA disagreers, e.g. GRK2 / CK1 / GSK3B).

Every (kinase, unit) score is persisted long, so "which kinases improve" and any n_sites cutoff is a
groupby, not a re-score. best-of-k is optimistic; each cluster already has >= MIN_CLUSTER sites
(motif_01c), and the summary keeps the best cluster's n_sites so a size filter is trivial.

Inputs   out/cddm_kmeans.parquet (motif_01c); kdata: cddm, pspa_scale, kinase_info;
         motif_16.recover / average_precision
Outputs  out/stratify_pspa.csv (long: kinase x unit x {ap, pearson, n_sites}),
         out/stratify_pspa_summary.csv (per kinase: overall / best-cluster / delta, both metrics),
         fig/stratify_pspa_{ap,pearson}.svg, fig/stratify_pspa_delta_{ap,pearson}.svg,
         fig/stratify_top_<group>_<kinase>.svg (top improver per group),
         fig/stratify_<kinase>_k<k>.svg (paper Fig. 4 c-f right column: CK1E/GSK3B/ERK1/ABL1 chosen
             stratification logos, PAPER_LOGOS; the left-column PSPA logo+heatmap is the pspa_logohm_<kinase>.svg pair)

Run:  python nbs/motif_25_stratify_pspa.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from motif_16_compare_methods import average_precision, combine_priming, recover
from paths import FIG, OUT

import kdata
from katlas.utils import group_color
from kplot.bar import plot_group_violin
from kplot.utils import paper_panel, save_svg, set_sns

KM = OUT / 'cddm_kmeans.parquet'
META = ['kinase', 'cluster', 'acceptor', 'n_sites', 'best_k', 'ch', 'weight', 'bic']


def _flank_pearson(a, b):
    "Whole-flank Pearson between two recovered PSSMs (shared cells, position 0 excluded)."
    cols = [c for c in a.columns if isinstance(c, (int, np.integer)) and c != 0 and c in b.columns]
    av, bv = a[cols].stack(), b[cols].stack()
    idx = av.index.intersection(bv.index)
    return float(np.corrcoef(av[idx].values, bv[idx].values)[0, 1]) if len(idx) > 2 else np.nan


def build_long(topk=5):
    "Per (kinase, unit) agreement with PSPA, unit = 'overall' (whole CDDM) or a k-means cluster id."
    "`topk` = PSPA-hotspot set size for AP (AP@5 default; AP@3 = topk=3)."
    km = pd.read_parquet(KM)
    pcols = [c for c in km.columns if c not in META]
    cddm = kdata.load('cddm')
    pspa = kdata.load('pspa_scale')
    pspa = pspa[~pspa.index.str.contains('_TYR')]
    grp = kdata.load('kinase_info').drop_duplicates('kinase').set_index('kinase').group
    ov_n = km.groupby('kinase').n_sites.sum().to_dict()                 # overall n = sum over clusters

    rows = []
    for k in sorted(set(km.kinase) & set(cddm.index) & set(pspa.index)):
        ref = recover(pspa.loc[k])
        ov = recover(cddm.loc[k])
        rows.append((k, grp.get(k), 'overall', int(ov_n.get(k, 0)),
                     average_precision(ov, ref, k=topk), _flank_pearson(ov, ref)))
        for _, r in km[km.kinase == k].iterrows():
            cm = recover(r[pcols].astype(float))
            rows.append((k, grp.get(k), str(int(r.cluster)), int(r.n_sites),
                         average_precision(cm, ref, k=topk), _flank_pearson(cm, ref)))
    return pd.DataFrame(rows, columns=['kinase', 'group', 'unit', 'n_sites', 'ap', 'pearson'])


def summarize(long):
    "Per kinase: overall vs best-cluster (max) for each metric, with the winning cluster + its n_sites."
    out = []
    for k, d in long.groupby('kinase'):
        ov = d[d.unit == 'overall'].iloc[0]
        cl = d[d.unit != 'overall']
        rec = {'kinase': k, 'group': ov.group}
        for m in ('ap', 'pearson'):
            best = cl.loc[cl[m].idxmax()] if len(cl) else ov
            rec[f'ov_{m}'] = ov[m]
            rec[f'best_{m}'] = best[m]
            rec[f'd_{m}'] = best[m] - ov[m]
            rec[f'best_{m}_cluster'] = best.unit
            rec[f'best_{m}_n'] = int(best.n_sites)
        out.append(rec)
    return pd.DataFrame(out)


def plot_scatter(summ, metric, label):
    "Overall vs best-cluster agreement per kinase; above the diagonal = stratification helped."
    o, b, d = summ[f'ov_{metric}'], summ[f'best_{metric}'], summ[f'd_{metric}']
    fig, ax = plt.subplots(figsize=paper_panel(1 / 3, ratio=1.0))
    lo = float(min(o.min(), b.min())); hi = float(max(o.max(), b.max()))
    ax.plot([lo, hi], [lo, hi], color='0.6', lw=.6, ls='--', zorder=0)
    for g in [g for g in group_color if g in set(summ.group)]:
        s = summ[summ.group == g]
        ax.scatter(s[f'ov_{metric}'], s[f'best_{metric}'], s=6, color=group_color[g],
                   edgecolor='none', alpha=.8)
    for _, r in summ.nlargest(6, f'd_{metric}').iterrows():                # label the biggest improvers
        ax.annotate(r.kinase, (r[f'ov_{metric}'], r[f'best_{metric}']), fontsize=5, color='0.25',
                    xytext=(2, 2), textcoords='offset points')
    ax.set_xlabel(f'overall CDDM vs PSPA ({label})', labelpad=2)
    ax.set_ylabel(f'best sub-motif vs PSPA ({label})', labelpad=2)
    ax.tick_params(length=2, pad=1.5)
    ax.text(.05, .95, f'improved: {int((d > 0).sum())}/{len(d)}', transform=ax.transAxes,
            fontsize=6, va='top')
    ax.spines[['top', 'right']].set_visible(False)
    save_svg(FIG / f'stratify_pspa_{metric}.svg')
    plt.close(fig)


def _improvers(summ, metric, min_n=20):
    "Kinases whose best sub-cluster beats the whole CDDM (delta > 0) on a cluster of >= min_n sites."
    return summ[(summ[f'd_{metric}'] > 0) & (summ[f'best_{metric}_n'] >= min_n)]


def plot_delta_violin(summ, metric, label, min_n=20):
    "Per-group violin of the improvement (delta), over the kinases that improve (delta > 0), group colors."
    d = _improvers(summ, metric, min_n).rename(columns={f'd_{metric}': 'delta'})
    order = d.groupby('group').delta.median().sort_values(ascending=False).index.tolist()   # big -> small
    fig, ax = plt.subplots(figsize=paper_panel(2 / 3, ratio=2.0))
    plot_group_violin(d, 'delta', 'group', hue=None, palette=group_color, order=order, ax=ax,
                      dot_size=2.5, violin_alpha=.35)
    ax.set_ylabel(f'Δ {label} (best sub-motif − overall)', labelpad=2)
    ax.tick_params(length=2, pad=1.5)
    ax.set_xticklabels(order, rotation=35, ha='right')
    ax.set_title(f'Stratification improvement by group (positive Δ, n={len(d)})', fontsize=8)
    save_svg(FIG / f'stratify_pspa_delta_{metric}.svg')
    plt.close(fig)


def plot_top_per_group_logos(summ, metric='ap', window=5, per_logo=0.75, min_n=20):
    "K-means stratification logo of each group's biggest improver (max delta on `metric`); title 'K (group)'."
    from motif_01d_cddm_plot_example import plot_kmeans_stratification
    imp = _improvers(summ, metric, min_n)                                 # improvers on a >= min_n cluster
    n_cl = pd.read_parquet(KM).groupby('kinase').cluster.nunique()        # clusters per kinase
    w = paper_panel(1 / 3, ratio=1.05)[0]                                 # 1/3-panel width; height scales
    top = (imp.loc[imp.groupby('group')[f'd_{metric}'].idxmax()]          # with the panel count (overall +
           .sort_values(f'd_{metric}', ascending=False))                 # clusters) so logos aren't squished
    for _, r in top.iterrows():
        n_panels = 1 + int(n_cl.get(r.kinase, 1))
        fig = plot_kmeans_stratification(r.kinase, window=window, figsize=(w, per_logo * n_panels),
                                         title=f'{r.kinase} ({r.group})')
        save_svg(FIG / f'stratify_top_{r.group}_{r.kinase}.svg')
        plt.close(fig)
    print('\ntop improver per group (by d_' + metric + '), stratification logos drawn:')
    print(top[['group', 'kinase', f'd_{metric}', f'best_{metric}_cluster',
               f'best_{metric}_n']].round(2).to_string(index=False))
    return top


def plot_top_n_logos(summ, metric='ap', n=10, window=5, per_logo=0.75, min_n=20):
    """The top-`n` improvers (by delta on `metric`) as k-means stratification logos; the winning
    cluster's per-logo title is bolded. Title 'K (group)'; height scales with the cluster count.
    Files are prefixed by rank so they sort in order: fig/stratify_rankNN_<kinase>.svg."""
    from motif_01d_cddm_plot_example import plot_kmeans_stratification
    imp = _improvers(summ, metric, min_n).sort_values(f'd_{metric}', ascending=False).head(n)
    n_cl = pd.read_parquet(KM).groupby('kinase').cluster.nunique()
    w = paper_panel(1 / 3, ratio=1.05)[0]
    for rank, (_, r) in enumerate(imp.iterrows(), 1):
        fig = plot_kmeans_stratification(r.kinase, window=window,
                                         figsize=(w, per_logo * (1 + int(n_cl.get(r.kinase, 1)))),
                                         title=f'{r.kinase} ({r.group})',
                                         bold_cluster=r[f'best_{metric}_cluster'])   # PSPA-matching winner
        save_svg(FIG / f'stratify_rank{rank:02d}_{r.kinase}.svg')
        plt.close(fig)
    print(f'\ntop {n} logos (winning cluster bolded):')
    print(imp[['kinase', 'group', f'd_{metric}', f'best_{metric}_cluster',
               f'best_{metric}_n']].round(3).to_string(index=False))
    return imp


#: the paper's stratification-logo panels: (kinase, forced k, bold cluster | None = auto best-PSPA)
PAPER_LOGOS = [('CK1E', 2, 0), ('GSK3B', 2, 0), ('ERK1', 3, None), ('ABL1', 3, None)]


def plot_paper_stratification(specs=PAPER_LOGOS, window=5, per_logo=0.75):
    """Paper stratification logos for explicit (kinase, k, bold) specs. Re-clusters CDDM into exactly
    k k-means sub-motifs and titles it 'kinase (group), k=k'; the bolded cluster is `bold` when given,
    else the cluster that best matches PSPA (AP@5). Height scales with k. Saves fig/stratify_<kinase>_k<k>.svg."""
    from motif_01d_cddm_plot_example import _kmeans_forced, plot_kmeans_stratification
    grp = kdata.load('kinase_info').drop_duplicates('kinase').set_index('kinase').group
    pspa = kdata.load('pspa_scale')
    pspa = pspa[~pspa.index.str.contains('_TYR')]
    w = paper_panel(1 / 3, ratio=1.05)[0]
    for kinase, k, bold in specs:
        if bold is None and kinase in pspa.index:                   # default: bold the cluster closest to PSPA
            ref = recover(pspa.loc[kinase])                         # combine_priming: _kmeans_forced returns a
            aps = {int(lbl.split()[1]): average_precision(combine_priming(m), ref)   # raw aa x position matrix,
                   for lbl, m in _kmeans_forced(kinase, k)}         # so it must be collapsed to the shared
                                                                     # alphabet before scoring against PSPA
            bold = max(aps, key=lambda c: (aps[c] if aps[c] == aps[c] else -1))
        fig = plot_kmeans_stratification(kinase, k=k, window=window, figsize=(w, per_logo * (1 + k)),
                                         title=f'{kinase} ({grp.get(kinase)}), k={k}', bold_cluster=bold)
        save_svg(FIG / f'stratify_{kinase}_k{k}.svg')
        plt.close(fig)
        print(f'{kinase} ({grp.get(kinase)}) k={k}: bold=cluster {bold}')


#: the paper's PSPA reference panels (same kinases as PAPER_LOGOS)
PAPER_PSPA = ['CK1E', 'GSK3B', 'ERK1', 'ABL1']


def plot_pspa_logo_heatmap(kinase, frac=1 / 3, height=4.0, hspace=0.02, height_ratios=(1.55, 5)):
    """One kinase's PSPA logo+heatmap (web `plot_logo_heatmap_pspa`, log2 value/median). Panel width =
    `frac` of the journal figure; `square=False` so the heatmap is flattened (non-square cells) and the
    height is set directly; `height_ratios` is the logo:heatmap split (a slightly taller logo than the
    stock 1:5). Title 'kinase - PSPA' (tight), short log2 y-axis + colorbar labels, all residues labeled,
    position numbers under the heatmap so the logo sits flush on top (no gap). Works for S/T (10
    positions) and TK (11) at the same width. Saves fig/pspa_logohm_<kinase>.svg."""
    from katlas.plot import style_logo_heatmap
    from katlas.pspa import plot_logo_heatmap_pspa
    from katlas.pssm import recover_pssm
    row = kdata.load('pspa').loc[kinase]
    positions = list(recover_pssm(row.dropna()).columns)                 # -5..+4 (S/T) or -5..+5 (TK)
    plot_logo_heatmap_pspa(row, title=f'{kinase} - PSPA', figsize=(paper_panel(frac)[0], height),
                           hspace=hspace, colorbar_title='log2', square=False, height_ratios=height_ratios)
    style_logo_heatmap(plt.gcf(), positions=positions, ylabel='log2', title=f'{kinase} - PSPA')
    save_svg(FIG / f'pspa_logohm_{kinase}.svg')
    plt.close('all')
    print(f'{kinase} - PSPA logo+heatmap')


def plot_paper_pspa(kinases=PAPER_PSPA):
    "The paper's PSPA logo+heatmap panels for the stratification kinases."
    for k in kinases:
        plot_pspa_logo_heatmap(k)


def main():
    set_sns()
    long = build_long()
    long.round(4).to_csv(OUT / 'stratify_pspa.csv', index=False)
    summ = summarize(long)
    summ.round(4).to_csv(OUT / 'stratify_pspa_summary.csv', index=False)
    print(f'{summ.kinase.nunique()} kinases scored')
    for m, lab in [('ap', 'AP@5'), ('pearson', 'flank Pearson')]:
        imp = _improvers(summ, m).sort_values(f'd_{m}', ascending=False)   # positive delta, best cluster n>=20
        print(f'\n== {lab}: {len(imp)}/{len(summ)} improve via stratification (best-cluster n>=20) '
              f'| mean delta {summ[f"d_{m}"].mean():+.3f} ==')
        print(imp.head(30)[['kinase', 'group', f'ov_{m}', f'best_{m}', f'd_{m}',
                            f'best_{m}_cluster', f'best_{m}_n']].round(3).to_string(index=False))
        plot_scatter(summ, m, lab)
        plot_delta_violin(summ, m, lab)
    plot_top_per_group_logos(summ, 'ap')
    plot_paper_stratification()
    plot_paper_pspa()
    print('\nwrote out/stratify_pspa.csv, out/stratify_pspa_summary.csv, fig/stratify_pspa_{ap,pearson}.svg, '
          'fig/stratify_<kinase>_k<k>.svg (paper logos)')


if __name__ == '__main__':
    main()
