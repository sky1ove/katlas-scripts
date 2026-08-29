"""scoring_02c · PSPA window sweep — the control for the CDDM-PSSM decline.

scoring_02b shows the generative CDDM PSSM getting *worse* with wider windows. The proposed
explanation is spurious hard zeros: a frequency matrix built from finite counts has cells no
substrate ever visited, and each one drags the multiply score down. PSPA is experimental — every
residue is measured at every position, so there are no hard zeros — and should therefore plateau
rather than decline. If it declines too, the explanation is wrong.

PSPA has no training, so there is nothing to rebuild per window; the same fixed matrix is simply
sliced. Its informative range is −5…+4 (position +5 is essentially empty).

One adjustment: the stored central S/T is MAX-normalised, so ~151 Ser/Thr kinases tie at
`0s = 1.0` and the w=0 (centre-only) point would be meaningless. It is replaced with the
continuous ratio S/(S+T), matching `get_logo`. This barely moves ±1 and wider (full AUCDF +0.003)
and leaves Tyr untouched.

Inputs   out/scoring_split.parquet, out/scoring_pool.parquet, kdata.load('pspa')
Outputs  out/scoring_pairs/window_pspa_pairs.parquet, fig/window_pspa_sweep.svg

Run:  python nbs/scoring_02c_window_pspa.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import scoring_util as su
from paths import FIG

import kdata
from katlas.scoring import multiply

#: (label, lo, hi) - PSPA's measured flank is -5..+4; +5 is ~empty and excluded.
#: These label strings are the join key scoring_04d maps onto its numeric window axis - keep them.
WINS = [('±0', 0, 0), ('±1', -1, 1), ('±2', -2, 2), ('±3', -3, 3), ('±4', -4, 4),
        ('full(-5..+4)', -5, 4)]
VAL_FRAC = 0.2


def continuous_center(pspa):
    "Replace the MAX-normalised central S/T with the continuous ratio S/(S+T); Tyr untouched."
    pspa = pspa.copy()          # do not mutate the cached matrix
    den = pspa['0s'] + pspa['0t']
    m = den > 0
    s, t = pspa['0s'] / den, pspa['0t'] / den
    for c in ['0s', '0S']:
        pspa.loc[m, c] = s[m]
    for c in ['0t', '0T']:
        pspa.loc[m, c] = t[m]
    print(f'central S/T made continuous for {int(m.sum())} kinases')
    return pspa


def sweep(split, pspa, site_kin, pools, common):
    res = []
    for seed in su.SEEDS:
        print(f'seed {seed}')
        tr_split = split[~split[f'test_{seed}']]
        vmask = su.strat_group_split(tr_split, frac=VAL_FRAC, seed=1000 + seed)
        valid = tr_split[vmask]
        valid = valid[valid.kinase_protein.isin(common)]
        test = split[split[f'test_{seed}']]
        test = test[test.kinase_protein.isin(common)]

        for bi, branch in enumerate(['ST', 'Tyr']):
            pool = pools[branch]
            va = su.branch_split(valid)[bi]
            va = va[va.kinase_protein.isin(pool)]
            te = su.branch_split(test)[bi]
            # Tyr's centre is a single Y, so a centre-only window carries no information
            wins = WINS if branch == 'ST' else [w for w in WINS if w[0] != '±0']
            for lab, lo, hi in wins:
                pspa_w = pspa[[c for c in pspa.columns if lo <= int(c[:-1]) <= hi]]
                for sp, ev in [('val', va), ('test', te)]:
                    sc = su.gen_scores(ev, pspa_w, multiply, pool)
                    p = su.score_pairs(ev, sc, pool, site_kin, 'PSPA', seed, branch)
                    p['window'], p['split'] = lab, sp
                    res.append(p)
                print(f'  {branch} {lab:14s} ({pspa_w.shape[1]} cols)')
    return pd.concat(res, ignore_index=True)


def main():
    split = su.load_split()
    pspa = continuous_center(kdata.load('pspa'))
    site_kin = su.site_kinase_map(split)
    pools, _ = su.load_pool()
    common = pools['ST'] + pools['Tyr']

    pairs = sweep(split, pspa, site_kin, pools, common)
    out = su.RES / 'window_pspa_pairs.parquet'
    pairs.to_parquet(out)
    print('saved', out, pairs.shape, '| splits:', sorted(pairs.split.unique()))

    su.plot_window_sweep(
        pairs, FIG / 'window_pspa_sweep.svg',
        'PSPA — window sweep: VALIDATION (top) vs TEST (bottom). mean ± std, 3 seeds '
        '(experimental matrix, no hard zeros)',
        'PSPA window', order=[w[0] for w in WINS])


if __name__ == '__main__':
    main()
