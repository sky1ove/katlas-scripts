"""scoring_03c_numkin_tradeoff · Rationalize the training num_kin cutoff (three-way trade-off).

As the training site-promiscuity cutoff (num_kin <= c) loosens, three quantities move in tension, and
one figure makes the sweet spot visible:

  1. # kinases retained    - pool kinases that still have enough training sites (>= MIN_SITES) once
                             the promiscuous ones are removed.
  2. recall@10 (validation) - the discriminative CDDM-seq MLP's benefit from more training sites
                             (from scoring_03a's persisted validation sweep).
  3. CDDM vs PSPA AP@5      - motif quality: does folding in promiscuous sites degrade the per-kinase
                             PSSM relative to the PSPA reference? Built here by rebuilding the CDDM
                             PSSM at each cutoff (S/T kinases), on a fixed kinase set for a fair curve.

The shipped training cutoff (num_kin <= 40) retains nearly complete kinase coverage while recall@10
and motif quality remain strong.

Inputs   out/scoring_pairs/numkin_sweep_pairs.parquet (03a), out/scoring_split.parquet, kdata pspa
Outputs  fig/numkin_kinases.svg, fig/numkin_recall_{macro,micro}.svg, fig/numkin_ap.svg (standalone panels)
Run:  python nbs/scoring_03c_numkin_tradeoff.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kd_util
import kdata
import numpy as np
import pandas as pd
import scoring_util as su
import seaborn as sns
from matplotlib import pyplot as plt
from paths import FIG

from kplot.utils import paper_panel, save_svg, set_sns

CUTOFFS = [5, 10, 20, 40, 80, 160]                        # finite training cutoffs shown (clean doublings)
LABELS = [str(c) for c in CUTOFFS] + ['all']              # x tick labels (sweep also has 'all')
ALLCAP = 10 ** 9                                          # 'all' = no cap
ALL_X = 320                                              # numeric x for the 'all' point (one doubling past 160)
XVAL = CUTOFFS + [ALL_X]                                  # numeric positions on a log2 axis (even doublings)
CHOSEN = 40                                               # shipped training cutoff
BR_COL = {'ST': '#3b6fb0', 'Tyr': '#e07b39'}             # branch colors (S/T blue, Tyr orange)


def recall_curve(split='val', macro=True):
    "recall@10 vs training cutoff, per branch, from 03a's persisted sweep (macro = per-kinase mean)."
    sw = pd.read_parquet(su.RES / 'numkin_sweep_pairs.parquet')
    sw = sw[sw.split == split]
    if macro:
        perk = su.summarize(sw, ['train_numkin', 'branch', 'kinase'])     # per-kinase recall@10
        m = (perk.groupby(['train_numkin', 'branch'])['top10'].mean()     # macro = mean over kinases
             .unstack('branch').reindex(LABELS))
    else:
        m = (su.summarize(sw, ['train_numkin', 'branch'])                 # micro = pooled over pairs
             .pivot_table(index='train_numkin', columns='branch', values='top10').reindex(LABELS))
    n = sw.groupby('branch').kinase.nunique().to_dict()                   # kinases per branch
    return m, n                                           # rows = cutoff label, cols = branch


def kinase_curve():
    "# pool kinases with >= MIN_SITES training sites at each cutoff, per branch."
    sp = su.load_split()
    pools, _ = su.load_pool()
    out = {}
    for br, pool in [('ST', set(pools['ST'])), ('Tyr', set(pools['Tyr']))]:
        d = (su.branch_split(sp)[0] if br == 'ST' else su.branch_split(sp)[1])
        d = d[d.kinase_protein.isin(pool)]
        counts = []
        for c in CUTOFFS + [ALLCAP]:
            vc = d[d.num_kin <= c].kinase_protein.value_counts()
            counts.append(int((vc >= su.MIN_SITES).sum()))
        out[br] = counts
    return pd.DataFrame(out, index=LABELS)


def ap_curve():
    """CDDM vs PSPA AP@5 vs cutoff, per branch, on the FULL candidate pool (219 S/T, 76 Tyr).

    CDDM PSSMs are built from all available sites at each num_kin cutoff (no minimum-count gate). The
    average is taken over the whole pool (a fixed reference set, so cutoffs are comparable); a kinase
    with too few sites for a PSSM at a strict cutoff contributes NaN there (dropped by the nanmean),
    so `neff` records how many of the pool are actually defined per cutoff.
    """
    from stats_util import boot_ci
    sp = su.load_split()
    pools, _ = su.load_pool()
    pspa = kdata.load('pspa')
    branch_df = {'ST': su.branch_split(sp)[0], 'Tyr': su.branch_split(sp)[1]}
    out = {}
    for br in ['ST', 'Tyr']:
        d = branch_df[br]
        kin = sorted(set(pools[br]) & set(pspa.index))                 # fixed = the full pool branch
        cols0 = su.build_pssm(d, pools[br]).columns
        cells = [c for c in pspa.columns if c in cols0 and int(c[:-1]) != 0 and c[-1] != 's']
        rows = {}
        for lab, cut in zip(LABELS, CUTOFFS + [ALLCAP]):
            P = su.build_pssm(d[d.num_kin <= cut], pools[br])
            rows[lab] = np.array([kd_util.nan_average_precision(pspa.loc[k, cells].to_numpy(float),
                                  P.loc[k, cells].to_numpy(float), k=5) for k in kin])
        mat = pd.DataFrame(rows, index=kin)                            # pool kinase x cutoff
        neff = mat.notna().sum()
        ci = {lab: (boot_ci(mat[lab].dropna().to_numpy()) if neff[lab] >= 3 else (np.nan, np.nan))
              for lab in LABELS}
        out[br] = {'mean': mat.mean(), 'ci': pd.DataFrame(ci, index=['lo', 'hi']).T,
                   'n': len(kin), 'neff': neff}
    return out


def _xaxis(ax):
    "log2 numeric x-axis (each step ~ one doubling), reversed so 'all' (loosest) sits on the left."
    ax.set_xscale('log', base=2)
    ax.set_xticks(XVAL)
    ax.set_xticklabels(LABELS)
    ax.minorticks_off()
    ax.invert_xaxis()                                                       # all ... 40 ... 5 (left to right)
    ax.axvline(CHOSEN, ls='--', color='0.6', lw=0.6)                        # chosen cutoff
    ax.set_xlabel('num_kin cutoff', labelpad=1.5)
    ax.tick_params(length=2, pad=1.5)
    ax.grid(alpha=0.3, lw=0.4)


def panel_kinases(nkin, fname, frac=1 / 2, ratio=1.7):
    "Standalone panel: # kinases with >= MIN_SITES sites surviving each num_kin filter, per branch."
    fig, ax = plt.subplots(figsize=paper_panel(frac, ratio=ratio))
    for br in ['ST', 'Tyr']:
        ax.plot(XVAL, nkin[br], '-o', color=BR_COL[br], ms=3, lw=1.0, label=br)
    ax.set_ylabel(f'# kinases (≥ {su.MIN_SITES} sites)', labelpad=1.5)
    ax.set_title('# kinases available for CDDM (with count cutoff), after num_kin filter', fontsize=8)
    ax.legend(loc='upper right', frameon=False, handlelength=1.4, labelspacing=0.25)
    _xaxis(ax)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.4)
    save_svg(fname)
    plt.close('all')
    print('  wrote', fname)


def panel_recall(rec, n, fname, kind='macro', frac=1 / 2, ratio=1.7):
    "Standalone panel: MLP recall@10 (macro or micro) on validation vs training num_kin cutoff, per branch."
    fig, ax = plt.subplots(figsize=paper_panel(frac, ratio=ratio))
    for br in ['ST', 'Tyr']:
        ax.plot(XVAL, rec[br], '-o', color=BR_COL[br], ms=3, lw=1.0, label=f"{br} (n={n[br]})")
    ax.set_ylabel(f'{kind} recall@10 (validation)', labelpad=1.5)
    ax.set_title(f'CDDM-seq MLP {kind} recall@10 vs training num_kin cutoff', fontsize=8)
    ax.legend(loc='upper right', frameon=False, handlelength=1.4, labelspacing=0.25)
    _xaxis(ax)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.4)
    save_svg(fname)
    plt.close('all')
    print('  wrote', fname)


def panel_ap(ap, fname, frac=1 / 3, ratio=1.15):
    "Standalone panel: CDDM vs PSPA AP@5 vs cutoff, per branch, over the full pool (same x-range)."
    fig, axx = plt.subplots(figsize=paper_panel(frac, ratio=ratio))
    for br in ['ST', 'Tyr']:
        d = ap[br]
        axx.fill_between(XVAL, d['ci']['lo'], d['ci']['hi'], color=BR_COL[br], alpha=0.15, linewidth=0)
        axx.plot(XVAL, d['mean'], '-o', color=BR_COL[br], ms=3, lw=1.0, label=f"{br} (n={d['n']})")
    axx.set_ylabel('CDDM vs PSPA AP@5', labelpad=1.5)
    axx.set_title('CDDM motif (no count cutoff) vs PSPA', fontsize=8)
    axx.legend(loc='upper right', frameon=False, handlelength=1.4, labelspacing=0.25)
    _xaxis(axx)
    sns.despine(ax=axx)
    plt.tight_layout(pad=0.4)
    save_svg(fname)
    plt.close('all')
    print('  wrote', fname)


def main():
    set_sns()
    macro, rec_n = recall_curve('val', macro=True)
    micro, _ = recall_curve('val', macro=False)
    panel_kinases(kinase_curve(), FIG / 'numkin_kinases.svg')
    panel_recall(macro, rec_n, FIG / 'numkin_recall_macro.svg', kind='macro')
    panel_recall(micro, rec_n, FIG / 'numkin_recall_micro.svg', kind='micro')
    ap = ap_curve()
    for br in ['ST', 'Tyr']:
        print(f'{br} AP@5 neff per cutoff:', ap[br]['neff'].to_dict())
    panel_ap(ap, FIG / 'numkin_ap.svg')


if __name__ == '__main__':
    main()
