"""scoring_02a · How much sequence context does the discriminative MLP need?

Train the CDDM-seq MLP at one-sided window half-widths w ∈ WINDOWS (positions −w…+w). Only the
one-hot position set is narrowed; everything else matches the shipped model in scoring_04c.

**The window is selected on a validation split, never on test.** Each seed carves a stratified
group 20% validation set out of its TRAIN data (same helper as the main split, so the carve is
leak-free too); the MLP trains on the inner-train and the sweep is scored on that held-out
validation. The test set is scored from the same model as a generalization check, tagged with a
`split` column so the figures never need a re-run, but it plays no part in the choice.

The selected window is written to a file that scoring_04c reads.

Inputs   out/scoring_split.parquet, out/scoring_pool.parquet, out/scoring_cddm_seed0.parquet (key space), pspa
Outputs  out/scoring_pairs/window_mlp_pairs.parquet, out/best_window_mlp.txt,
         fig/window_mlp_sweep.svg

Run:  python nbs/scoring_02a_window_mlp.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import scoring_util as su
from paths import FIG

WINDOWS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 15, 20]   # w=0 = centre residue only
VAL_FRAC = 0.2
TRAIN_CAP = 70000


def sweep(split, site_kin, pools, common, cols):
    res = []
    for seed in su.SEEDS:
        print(f'seed {seed}')
        tr_split = split[~split[f'test_{seed}']]                     # test held out entirely
        vmask = su.strat_group_split(tr_split, frac=VAL_FRAC, seed=1000 + seed)
        inner, valid = tr_split[~vmask], tr_split[vmask]
        site_kin_inner = su.site_kinase_map(inner)                   # mask from inner-train only
        inner = inner[(inner.num_kin <= su.BASE_NUMKIN) & (inner.kinase_protein.isin(common))]
        valid = valid[valid.kinase_protein.isin(common)]
        test = split[split[f'test_{seed}']]
        test = test[test.kinase_protein.isin(common)]

        for bi, branch in enumerate(['ST', 'Tyr']):
            pool = pools[branch]
            pidx = {k: i for i, k in enumerate(pool)}
            tr = su.branch_split(inner)[bi]
            tr = tr[tr.kinase_protein.isin(pool)].sample(n=min(TRAIN_CAP, len(tr)), random_state=seed)
            va = su.branch_split(valid)[bi]
            va = va[va.kinase_protein.isin(pool)]
            te = su.branch_split(test)[bi]
            ytr = np.array([pidx[k] for k in tr.kinase_protein])

            # mask and class weights are window-independent, so build them once per branch
            mask = su.otk_mask(tr.site_seq.values, ytr, pool, site_kin_inner)
            cbal = su.class_balance(tr.kinase_protein, pool)

            for w in WINDOWS:
                cols_w = [c for c in cols if abs(int(c[:-1])) <= w]
                model, _ = su.fit_mlp(su.onehot(tr.site_seq.values, cols_w), ytr, len(pool),
                                      hidden=512, depth=2, seed=seed, mask=mask, class_weight=cbal,
                                      groups=tr.site_seq.str.upper().to_numpy())
                for sp, ev in [('val', va), ('test', te)]:
                    sc = su.mlp_scores(model, su.onehot(ev.site_seq.values, cols_w))
                    p = su.score_pairs(ev, sc, pool, site_kin, 'CDDM-seq MLP', seed, branch)
                    p['window'], p['split'] = w, sp
                    res.append(p)
                print(f'  {branch} w={w:>2} (±{w}, {2 * w + 1}-mer, D={len(cols_w)})')
    return pd.concat(res, ignore_index=True)


def main():
    split = su.load_split()
    site_kin = su.site_kinase_map(split)     # FULL labels - eval only
    pools, _ = su.load_pool()
    common = pools['ST'] + pools['Tyr']

    pairs = sweep(split, site_kin, pools, common, su.onehot_cols())
    out = su.RES / 'window_mlp_pairs.parquet'
    pairs.to_parquet(out)
    print('saved', out, pairs.shape, '| splits:', sorted(pairs.split.unique()))

    su.select_window(pairs, su.OUT / 'best_window_mlp.txt', 'MLP')
    su.plot_window_sweep(
        pairs, FIG / 'window_mlp_sweep.svg',
        'CDDM-seq MLP — window sweep: VALIDATION (top, selects the window) vs '
        'TEST (bottom, generalization). mean ± std, 3 seeds',
        'window half-width  w  (±w residues)', xticks=WINDOWS)


if __name__ == '__main__':
    main()
