"""kd_09 · The external test set: validate the predictions against non-human PhosphoSitePlus data.

kd_07's predictions are for kinases with no measured specificity — but some of them, in other
species, do have substrates recorded in PhosphoSitePlus. That is a genuinely **external test set**
the model never saw (different species, ground truth measured independently from real substrates, not
the training assay): build a PSSM straight from those PSP substrates and compare it to the prediction.

Only non-human organisms with enough records are used (the human motifs are what the model was
trained on), and a kinase needs at least MIN_SITES substrate sites before its empirical PSSM is
stable enough to compare against. **Scope caveat:** the kinases with enough non-human substrate data
are all near-identical orthologs of human kinases (every one falls below the confidence threshold,
max nn_dist ~4 « threshold), so this validates the *predictable* regime — it cannot test the threshold
location, and the genuinely novel dark-kinome predictions have no external ground truth. Each kinase's
gene name and nn_dist are reported alongside the agreement metrics.

Agreement uses the same three metrics as the rest of the pipeline — mean per-position Spearman, AP@5,
and overall Pearson — over the ±5 flank the two matrices share (the position-0 acceptor excluded).
The best and worst few are drawn side by side, PSP on the left and the prediction on the right.

**Unique-kinase reporting (avoid pseudoreplication).** Several species' orthologs map to one human
kinase (mouse+rat CAMK2A, cow+mouse+rat PRKACA, mouse+chicken+rat SRC), so the 35 species-kinase
pairs are only ~23 independent kinases. Counting the pairs as independent would inflate n and
double-weight the well-sampled kinases in the median. The headline is therefore reported at the
**unique human-kinase level** — a kinase's species are averaged first, then the median is taken over
unique kinases — with a per-Manning-group breakdown. The per-species table is still written for detail.

Inputs   raw/psp_ks_dataset_2408.csv, out/kd_pred_new_pspa.parquet (kd_07)
Outputs  out/kd_validation_nonhuman.csv (per-species), out/kd_validation_by_kinase.csv (unique
         kinase, ortholog-averaged), fig/kd_validate_<uniprot>.svg

Run:  python nbs/kd_09_validate_nonhuman.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kd_util
import numpy as np
import pandas as pd
from stats_util import nan_average_precision, nan_pearson, nan_spearman   # NaN-safe metric primitives
from matplotlib import pyplot as plt
from paths import FIG, OUT, RAW

from katlas.pssm import get_cluster_pssms, recover_pssm
from kplot.utils import save_svg, set_sns

MIN_ORGANISM = 13    # an organism needs more than this many records to be worth using
MIN_SITES = 30       # a kinase needs this many substrate sites for a stable empirical PSSM
N_SHOW = 5           # best / worst examples drawn
METHODS = [('pspa', 'PSPA'), ('cddm', 'CDDM'), ('mlp_attr', 'MLP-attr')]   # predicted methods to compare
ANNOT = ['nn_dist', 'nn_group', 'threshold', 'predictable', 'proximity_tier',   # non-PSSM columns
         'organism', 'is_human']   # (kd_06b/kd_07 annotations) dropped so only PSSM cells remain


def nonhuman_pssms():
    "Empirical PSSMs from the non-human PhosphoSitePlus records + a uniprot->gene-name map."
    df = pd.read_csv(RAW / 'psp_ks_dataset_2408.csv')
    df['kin_uniprot'] = df.KIN_ACC_ID.str.split('-').str[0]

    counts = df.KIN_ORGANISM.value_counts()
    # [1:] drops human, which is the training data
    organisms = counts[counts > MIN_ORGANISM].index[1:]
    df = df[df.KIN_ORGANISM.isin(organisms)]
    print(f'non-human organisms: {list(organisms)}')

    names = df.drop_duplicates('kin_uniprot').set_index('kin_uniprot')['GENE'].str.upper()  # display name
    df['sub_site'] = df.SUB_ACC_ID.str.split('-').str[0] + '_' + df.SUB_MOD_RSD
    df = df.drop_duplicates(subset=['kin_uniprot', 'sub_site'])
    df['site_seq'] = df['SITE_+/-7_AA']
    print(f'records after dedup: {len(df):,}')

    pssms = get_cluster_pssms(df, cluster_col='kin_uniprot', count_thr=MIN_SITES)
    print(f'kinases with >= {MIN_SITES} sites: {len(pssms)}')
    return pssms, names


def nn_dist_map():
    "Each non-human kinase's nn_dist + nearest-human-kinase group (from kd_07 predictions), keyed by UniProt."
    p = pd.read_parquet(OUT / 'kd_pred_new_pspa.parquet')
    p = p.assign(u=p.index.str.split('_').str[0])
    nearest = p.loc[p.groupby('u').nn_dist.idxmin()].set_index('u')   # the closest domain per UniProt
    return nearest.nn_dist, nearest.nn_group, float(p.threshold.iloc[0])


def score(psp, pred):
    """Per-kinase Spearman / AP@5 / overall Pearson over the shared FLANK cells (position 0 excluded).

    The PSP empirical PSSM (the reference) carries NaN cells (empty phospho-priming s/t/y positions), so
    every metric goes through the NaN-safe primitives (stats_util) — they mask cells that are NaN in
    either matrix before correlating/ranking. Without that a single NaN voids the whole pair (corr → NaN)
    and lets NaN sort into the AP@5 top-k; that silently dropped ABL1/CAMK2B/CDK1 in earlier versions."""
    cols = [c for c in sorted(set(psp.columns) & set(pred.columns))
            if int(re.match(r'(-?\d+)', str(c)).group(1)) != 0]
    bypos = {}
    for c in cols:
        bypos.setdefault(int(re.match(r'(-?\d+)', str(c)).group(1)), []).append(c)
    rows = {}
    for i in psp.index:
        a, b = psp.loc[i, cols].to_numpy(), pred.loc[i, cols].to_numpy()
        by = list(bypos.values())
        sp = np.nanmean([nan_spearman(psp.loc[i, cc], pred.loc[i, cc]) for cc in by if len(cc) >= 3])
        rows[i] = (sp, nan_average_precision(a, b, k=5), nan_pearson(a, b))   # empirical = reference
    return pd.DataFrame(rows, index=['spearman', 'ap', 'pearson']).T.sort_values('spearman', ascending=False)


def load_pred(target):
    "Predicted PSSM keyed by UniProt (drop kd_06b/kd_07 annotation columns, keep only PSSM cells)."
    p = pd.read_parquet(OUT / f'kd_pred_new_{target}.parquet').drop(columns=ANNOT, errors='ignore')
    p.index = p.index.str.split('_').str[0]             # kd_ID -> UniProt
    return p[~p.index.duplicated()]


def dedup_figure(res_all, uni, names, gene2group, n_species, metric='spearman', mlabel='flank Spearman'):
    """Visualise the ortholog collapse: (A) each unique kinase's per-species points averaged to one value,
    (B) raw vs unique medians barely move, (C) unique-kinase agreement by Manning group. `metric` is one of
    the three agreement scores (flank Spearman / AP@5 / Pearson); the CDDM method is shown in A/C as it
    validates best (same frequency basis). Saved as fig/kd_validation_dedup[_<metric>].svg."""
    import seaborn as sns
    methods, GREY, RED = ['PSPA', 'CDDM', 'MLP-attr'], '#9bb8cc', '#c0392b'
    r = res_all['CDDM'].assign(gene=res_all['CDDM'].index.map(names))
    ug = uni['CDDM'][metric].dropna().sort_values()          # unique (ortholog-averaged), best on top
    order = list(ug.index)

    fig = plt.figure(figsize=(13, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], height_ratios=[1, 1], wspace=0.28, hspace=0.5)
    axA, axB, axC = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    for i, g in enumerate(order):                            # A: species dots -> averaged unique dot
        pts = r.loc[r.gene == g, metric].dropna().to_numpy()
        axA.plot([pts.min(), pts.max()], [i, i], color='0.82', lw=1, zorder=1)
        axA.scatter(pts, [i] * len(pts), s=20, color=GREY, zorder=2,
                    label='per species' if i == 0 else None)
        axA.scatter([ug[g]], [i], s=58, color=RED, edgecolor='white', lw=0.6, zorder=3,
                    label='unique (ortholog-averaged)' if i == 0 else None)
        if int(n_species.get(g, 1)) > 1:                     # mark the multi-species (collapsed) kinases
            axA.annotate(f'×{int(n_species[g])}', (max(pts.max(), ug[g]) + 0.012, i), va='center',
                         fontsize=7, color='#6d818c')
    axA.axvline(ug.median(), color=RED, ls='--', lw=1, zorder=0)
    axA.text(ug.median(), len(order) - 0.4, ' unique median', color=RED, fontsize=7.5,
             va='top', ha='left', style='italic')
    axA.set_yticks(range(len(order)), order, fontsize=8)
    axA.set_ylim(-0.7, len(order) - 0.3)
    axA.set_xlabel(f'{mlabel} (CDDM vs PhosphoSitePlus)')
    axA.set_title(f'A · Ortholog collapse — {len(res_all["CDDM"])} species-pairs to {len(uni["CDDM"])} '
                  'unique kinases', fontsize=10)
    axA.legend(loc='lower right', fontsize=8, frameon=False)

    x, w = np.arange(len(methods)), 0.38                     # B: raw vs unique medians
    raw_med = [res_all[m][metric].median() for m in methods]
    uni_med = [uni[m][metric].median() for m in methods]
    axB.bar(x - w / 2, raw_med, w, color=GREY, label=f'per species (n={len(res_all["CDDM"])})')
    axB.bar(x + w / 2, uni_med, w, color=RED, label=f'unique (n={len(uni["CDDM"])})')
    for xi, (a, b) in enumerate(zip(raw_med, uni_med)):
        axB.text(xi - w / 2, a + 0.012, f'{a:.2f}', ha='center', fontsize=7.5)
        axB.text(xi + w / 2, b + 0.012, f'{b:.2f}', ha='center', fontsize=7.5)
    axB.set_xticks(x, methods)
    axB.set_ylabel(f'median {mlabel}')
    axB.set_ylim(0, max(raw_med + uni_med) * 1.22)
    axB.set_title('B · Medians barely move after collapse', fontsize=10)
    axB.legend(fontsize=7.5, frameon=False)

    dfg = pd.DataFrame({'v': ug.to_numpy(), 'group': [gene2group[g] for g in ug.index]})   # C: by group
    og = dfg.groupby('group').v.median().sort_values(ascending=False).index
    sns.boxplot(data=dfg, x='group', y='v', order=og, ax=axC, color='#d9e2ea', fliersize=0,
                width=0.6, linewidth=0.8)
    sns.stripplot(data=dfg, x='group', y='v', order=og, ax=axC, color='#2c3e50', size=3.5, alpha=0.6)
    axC.set_xticks(range(len(og)), [f'{g}\n(n={(dfg.group == g).sum()})' for g in og], fontsize=8.5)
    axC.set_xlabel('')
    axC.set_ylabel(mlabel)
    axC.set_title('C · Unique-kinase agreement by Manning group (CDDM)', fontsize=10)

    fig.suptitle(f'External validation ({mlabel}), corrected for pseudoreplication: multi-species orthologs '
                 'collapsed to unique human kinases', fontsize=11.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    suffix = '' if metric == 'spearman' else f'_{metric}'
    save_svg(FIG / f'kd_validation_dedup{suffix}.svg')
    plt.close('all')


def heatmap(ax, flat, title, signed):
    "One PSSM heatmap (residue x position); RdBu for signed methods, viridis for the distributions."
    m = recover_pssm(flat)
    v = np.nanmax(np.abs(m.to_numpy()))
    kw = dict(cmap='RdBu_r', vmin=-v, vmax=v) if signed else dict(cmap='viridis')
    ax.imshow(m.to_numpy(), aspect='auto', **kw)
    ax.set_yticks(range(len(m.index)), m.index, fontsize=5)
    ax.set_xticks(range(len(m.columns)), m.columns, rotation=90, fontsize=5)
    ax.set_title(title, fontsize=9)


def main():
    set_sns()
    preds = {name: load_pred(t) for t, name in METHODS}
    psp, names = nonhuman_pssms()
    nn, groups, threshold = nn_dist_map()

    res_all = {}
    for name, pred in preds.items():
        shared = sorted(set(pred.index) & set(psp.index))
        r = score(psp.loc[shared], pred.loc[shared])
        res_all[name] = r

    combined = pd.concat({n: r for n, r in res_all.items()}, axis=1)
    # prepend an `info` block: kinase gene, nearest-human-kinase group, nn_dist to the nearest labelled kinase
    info = pd.DataFrame({('info', 'kinase'): combined.index.map(names),
                         ('info', 'group'): combined.index.map(groups),
                         ('info', 'nn_dist'): combined.index.map(nn).round(2)}, index=combined.index)
    combined = pd.concat([info, combined], axis=1)
    combined.round(4).to_csv(OUT / 'kd_validation_nonhuman.csv')   # per-species (granular)

    # ---- collapse multi-species orthologs to unique human kinases (fix pseudoreplication) ----
    # several species' orthologs share one human gene symbol (mouse+rat CAMK2A, cow+mouse+rat PRKACA,
    # mouse+chicken+rat SRC): counting them as independent inflates n and double-weights well-sampled
    # kinases in the median. Report at the unique-kinase level — average a kinase's species FIRST, then
    # take the median over unique kinases (n = independent kinases, not species x kinase pairs).
    meta = pd.DataFrame({'gene': combined.index.map(names), 'group': combined.index.map(groups)},
                        index=combined.index)
    gene2group = meta.drop_duplicates('gene').set_index('gene')['group']
    n_species = meta.gene.value_counts()
    uni = {name: res_all[name].groupby(res_all[name].index.map(names))[['spearman', 'ap', 'pearson']].mean()
           for _, name in METHODS}

    print('\npseudoreplication check — per-species (raw) vs unique-kinase (ortholog-averaged) medians '
          '(95% CI = bootstrap over unique kinases, the independent unit):')
    for _, name in METHODS:
        rr, bg = res_all[name], uni[name]
        lo, hi = kd_util.boot_ci(bg.spearman.to_numpy(), np.median)                # Sp CI over unique kinases
        print(f'  {name:8}: raw n={len(rr):2d}  Sp {rr.spearman.median():.3f} / AP {rr.ap.median():.2f} / '
              f'Pe {rr.pearson.median():.2f}   ->   unique n={len(bg):2d}  Sp {bg.spearman.median():.3f} '
              f'[95% CI {lo:.2f}, {hi:.2f}] / AP {bg.ap.median():.2f} / Pe {bg.pearson.median():.2f}')

    uni_combined = pd.concat({name: uni[name] for _, name in METHODS}, axis=1)
    uni_info = pd.DataFrame({('info', 'group'): uni_combined.index.map(gene2group),
                             ('info', 'n_species'): uni_combined.index.map(n_species)}, index=uni_combined.index)
    uni_combined = pd.concat([uni_info, uni_combined], axis=1).sort_index()
    uni_combined.round(4).to_csv(OUT / 'kd_validation_by_kinase.csv')

    print('\nunique-kinase medians by Manning kinase group (CDDM, ortholog-averaged):')
    cddm_u = uni['CDDM'].assign(group=uni['CDDM'].index.map(gene2group))
    for g, sub in cddm_u.groupby('group'):
        print(f'  {g:5} n={len(sub):2d}  Spearman {sub.spearman.median():.3f} / AP {sub.ap.median():.2f}')

    dedup_figure(res_all, uni, names, gene2group, n_species, 'spearman', 'flank Spearman')
    dedup_figure(res_all, uni, names, gene2group, n_species, 'ap', 'AP@5')
    print(f'wrote {FIG / "kd_validation_dedup.svg"} and kd_validation_dedup_ap.svg (ortholog-collapse)')

    below = int((combined[('info', 'nn_dist')] <= threshold).sum())
    print(f'\nwrote {OUT / "kd_validation_nonhuman.csv"} (per-species, {len(combined)} pairs) and '
          f'{OUT / "kd_validation_by_kinase.csv"} ({len(uni_combined)} unique kinases). '
          f'{below}/{len(combined)} species-pairs below the {threshold:.1f} threshold — max nn_dist '
          f'{combined[("info", "nn_dist")].max():.1f}. This is an EXTERNAL test of the predictable '
          f'(below-threshold) regime: every validated kinase is a near-identical ortholog of a human one.')

    # comparison figures for EVERY validated kinase (there are only a few dozen), ranked by CDDM agreement
    rank = res_all['CDDM'].index
    for kin in rank:
        panels = [('PSP (observed)', psp.loc[kin], False)]
        panels += [(name, preds[name].loc[kin], name == 'MLP-attr') for _, name in METHODS
                   if kin in preds[name].index]
        fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 4.2))
        for ax, (title, flat, signed) in zip(np.atleast_1d(axes), panels):
            heatmap(ax, flat, title, signed)
        label = names.get(kin, kin)
        fig.suptitle(f'{label} ({kin}) — nn_dist {nn.get(kin, float("nan")):.1f}, '
                     f'CDDM ρ {res_all["CDDM"].spearman.get(kin, float("nan")):.2f}', fontsize=11)
        fig.tight_layout()
        save_svg(FIG / f'kd_validate_{kin}.svg')
        plt.close('all')
    print(f'wrote {len(rank)} comparison figures (PSP + {len(METHODS)} methods) to {FIG}')


if __name__ == '__main__':
    main()
