"""scoring_02b · The same window sweep for the generative CDDM PSSM.

Companion to scoring_02a. Score the CDDM PSSM using only positions −w…+w, over the same windows
and the same validation-based selection: the PSSM is rebuilt on the inner-train so validation is
genuinely held out, then scored on both validation and test.

Unlike the MLP, the generative PSSM *declines* with wider windows — a frequency matrix built from
finite substrate counts picks up spurious hard zeros at distant positions, and each one costs the
multiply score. scoring_02c runs PSPA through the same sweep as a control: PSPA is experimental and
has no hard zeros, so it should not decline.

The selected window is written to a file; scoring_04b reports on test at ±5.

Inputs   out/scoring_split.parquet, out/scoring_pool.parquet
Outputs  out/scoring_pairs/window_cddm_pairs.parquet, fig/window_cddm_sweep.svg

Run:  python nbs/scoring_02b_window_cddm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import scoring_util as su
from paths import FIG

from katlas.scoring import multiply_23

WINDOWS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 15, 20]   # w=0 = centre residue only
VAL_FRAC = 0.2


def sweep(split, site_kin, pools, common):
    res = []
    for seed in su.SEEDS:
        print(f'seed {seed}')
        tr_split = split[~split[f'test_{seed}']]                     # test held out entirely
        vmask = su.strat_group_split(tr_split, frac=VAL_FRAC, seed=1000 + seed)
        inner, valid = tr_split[~vmask], tr_split[vmask]
        # PSSM built on the inner-train only, so validation is genuinely held out
        cddm = su.build_pssm(inner[inner.num_kin <= su.BASE_NUMKIN], common)
        valid = valid[valid.kinase_protein.isin(common)]
        test = split[split[f'test_{seed}']]
        test = test[test.kinase_protein.isin(common)]

        for bi, branch in enumerate(['ST', 'Tyr']):
            pool = pools[branch]
            va = su.branch_split(valid)[bi]
            va = va[va.kinase_protein.isin(pool)]
            te = su.branch_split(test)[bi]
            for w in WINDOWS:
                cddm_w = su.win(cddm, w)
                for sp, ev in [('val', va), ('test', te)]:
                    sc = su.gen_scores(ev, cddm_w, multiply_23, pool)
                    p = su.score_pairs(ev, sc, pool, site_kin, 'CDDM', seed, branch)
                    p['window'], p['split'] = w, sp
                    res.append(p)
                print(f'  {branch} w={w:>2} (±{w}, {2 * w + 1}-mer, {cddm_w.shape[1]} PSSM cols)')
    return pd.concat(res, ignore_index=True)


def main():
    split = su.load_split()
    site_kin = su.site_kinase_map(split)
    pools, _ = su.load_pool()
    common = pools['ST'] + pools['Tyr']

    pairs = sweep(split, site_kin, pools, common)
    out = su.RES / 'window_cddm_pairs.parquet'
    pairs.to_parquet(out)
    print('saved', out, pairs.shape, '| splits:', sorted(pairs.split.unique()))

    su.select_window(pairs, 'PSSM')
    su.plot_window_sweep(
        pairs, FIG / 'window_cddm_sweep.svg',
        'Generative CDDM PSSM — window sweep: VALIDATION (top, selects the window) vs '
        'TEST (bottom). mean ± std, 3 seeds',
        'window half-width  w  (±w residues)', xticks=WINDOWS)


if __name__ == '__main__':
    main()
