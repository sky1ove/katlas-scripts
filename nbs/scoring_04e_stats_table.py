"""scoring_04e_stats_table · Supplementary statistics table for Figure 2.

Two tables, each on the reliably annotated num_kin <= 10 test subset, per branch (S/T, Tyr). Everything
derives from the persisted per-pair scores (never a re-score); confidence intervals resample the
biological unit, the KINASE (a cluster bootstrap), not the pseudoreplicated site.

  Sheet 'scores'            : the nine methods x {micro recall@10 (pooled over sites), macro recall@10
                              (averaged over kinases), AUCDF}, each mean + bootstrap 95% CI. Micro/AUCDF
                              via su.boot_micro_ci (kinase cluster bootstrap); macro via a bootstrap over
                              the per-kinase recall values.
  Sheet 'paired_vs_shipped' : the shipped model (CDDM-seq MLP + PSPA) vs each prespecified generative
                              baseline (raw CDDM, CDDM pct, PSPA, PSPA pct): median per-kinase
                              delta recall@10 + bootstrap 95% CI, paired Wilcoxon p, Holm-adjusted p.

Inputs   out/scoring_pairs/{pspa,cddm,cddm_seq}_pairs.parquet, out/scoring_split.parquet, scoring_pool
Outputs  out/scoring_fig2_stats.xlsx
Run:  python nbs/scoring_04e_stats_table.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import scoring_04d_figures as f4
import scoring_fig2_panels as f2
import scoring_util as su
from paths import OUT

SHIPPED = 'CDDM-seq MLP + PSPA'
BASELINES = ['CDDM', 'CDDM pct', 'PSPA', 'PSPA pct']    # the prespecified generative baselines
BRANCHES = [('ST', 'S/T'), ('Tyr', 'Tyr')]


def _macro_ci(vals, B=su.N_BOOT, seed=0):
    "Bootstrap 95% CI for macro recall@10 = resample kinases, re-average their per-kinase recall."
    v = np.asarray(vals, float)
    if len(v) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _holm(pvals):
    "Holm-Bonferroni step-down adjusted p-values."
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(run, 1.0)
    return adj


def scores_table(pairs):
    "Nine methods x three metrics, mean + bootstrap 95% CI, per branch."
    perk = su.summarize(pairs, ['method', 'branch', 'kinase'])
    rows = []
    for br, blab in BRANCHES:
        d = pairs[pairs.branch == br]
        agg = su.summarize(d, ['method', 'branch']).set_index('method')     # pooled (micro) point estimates
        n_windows = d.site_seq.nunique()                                    # unique sequence windows (site_seq)
        for m in f4.ALL:
            ci = su.boot_micro_ci(d[d.method == m])                         # kinase cluster bootstrap
            kv = perk[(perk.method == m) & (perk.branch == br)]['top10'].to_numpy()
            mlo, mhi = _macro_ci(kv)
            rows.append({
                'branch': blab, 'method': m, 'n_kinases': len(kv), 'n_windows': n_windows,
                'micro_recall@10': agg.loc[m, 'top10'], 'micro_lo': ci['top10'][0], 'micro_hi': ci['top10'][1],
                'macro_recall@10': float(kv.mean()), 'macro_lo': mlo, 'macro_hi': mhi,
                'AUCDF': agg.loc[m, 'AUCDF'], 'aucdf_lo': ci['AUCDF'][0], 'aucdf_hi': ci['AUCDF'][1]})
    return pd.DataFrame(rows)


def paired_table(pairs):
    "Shipped model vs each prespecified generative baseline, per branch, Holm-adjusted within a branch."
    perk = su.summarize(pairs, ['method', 'branch', 'kinase'])
    rows = []
    for br, blab in BRANCHES:
        wide = perk[perk.branch == br].pivot_table(index='kinase', columns='method', values='top10')
        recs = []
        for i, b in enumerate(BASELINES):
            x = wide[[SHIPPED, b]].dropna()
            rep = su.wilcoxon_report(x[SHIPPED].to_numpy(), x[b].to_numpy(), seed=i)
            recs.append((b, len(x), rep))
        holm = _holm([r[2]['p'] for r in recs])
        for (b, n, rep), ph in zip(recs, holm):
            lo, hi = rep['ci']
            rows.append({'branch': blab, 'comparison': f'{SHIPPED} vs {b}', 'n_kinases': n,
                         'median_delta_recall@10': rep['median_delta'], 'delta_lo': lo, 'delta_hi': hi,
                         'W': rep['W'], 'p_raw': rep['p'], 'p_holm': ph})
    return pd.DataFrame(rows)


def held_out_counts():
    """Exact held-out counts per branch, reproduced from the split + pool via the benchmark's own
    branch_split rule (kinase class AND central acceptor). Verified to match the scored pairs' site_seq
    exactly. Reports both the substrate-site count (sub_site = protein + position, the metric's unit and
    the unit of the promiscuity figure) and the unique-sequence-window count (site_seq)."""
    sp = su.load_split()
    pools, _ = su.load_pool()
    st, tyr = su.branch_split(sp)
    st = st[st.kinase_protein.isin(set(pools['ST']))]
    tyr = tyr[tyr.kinase_protein.isin(set(pools['Tyr']))]
    rows = []
    for br, d in [('S/T', st), ('Tyr', tyr)]:
        for cap, lab in [(None, 'all'), (su.NK_MAIN, f'num_kin<={su.NK_MAIN}')]:
            dd = d if cap is None else d[d.num_kin <= cap]
            rows.append({'branch': br, 'subset': lab, 'n_kinases': dd.kinase_protein.nunique(),
                         'n_sites': dd.sub_site.nunique(), 'n_windows': dd.site_seq.nunique()})
    return pd.DataFrame(rows)


def main():
    pairs = f2.load_pairs_full(su.NK_MAIN)
    counts = held_out_counts()
    scores = scores_table(pairs)
    paired = paired_table(pairs)
    print('\n== held-out counts ==\n', counts.to_string(index=False))

    num = scores.select_dtypes('number').columns.difference(['n_kinases', 'n_windows'])
    scores[num] = scores[num].round(4)
    for c in ['median_delta_recall@10', 'delta_lo', 'delta_hi']:
        paired[c] = paired[c].round(4)

    out = OUT / 'scoring_fig2_stats.xlsx'
    with pd.ExcelWriter(out) as xl:
        counts.to_excel(xl, sheet_name='held_out', index=False)
        scores.to_excel(xl, sheet_name='scores', index=False)
        paired.to_excel(xl, sheet_name='paired_vs_shipped', index=False)
    print('wrote', out)
    print('\n== scores (S/T) ==\n', scores[scores.branch == 'S/T'].to_string(index=False))
    print('\n== paired vs shipped ==\n', paired.to_string(index=False))


if __name__ == '__main__':
    main()
