"""scoring_03a · Does site promiscuity help or hurt? Two num_kin experiments.

`num_kin` is how many kinases are annotated for a site. Sites hit by many kinases are both
noisier labels and, arguably, less informative training examples — these two sweeps justify the
`num_kin ≤ 40` training cutoff used everywhere else.

1. **Training sweep** — retrain the CDDM-seq MLP with the training set capped at each cutoff in
   NUMKINS, and **select the cutoff on a held-out VALIDATION carve, never on test** (the same
   leak-free protocol the flank-window sweep uses in scoring_02a): a stratified group 20% validation
   set is carved out of train, the MLP fits on the inner-train, and each cutoff is scored on that
   validation set. The test set is scored from the same model as a generalization check, tagged with
   a `split` column, so the choice never touches test. The selected cutoff is written to a file.
2. **Test stratification** — fix the MLP (num_kin ≤ 40) and PSPA, score the whole test once, then
   bin the *test* sites by their own num_kin. This is descriptive (characterising test-site
   difficulty, selecting nothing), so it legitimately reports on test. Every pair carries its
   num_kin, so the binning is a downstream groupby rather than a re-score.

Single seed: the sets are large enough that the estimates are stable, and the sweep is 22 model fits.

Inputs   out/scoring_split.parquet, out/scoring_pool.parquet, out/scoring_cddm_seed0.parquet (key space), pspa
Outputs  out/scoring_pairs/numkin_sweep_pairs.parquet (val+test), out/scoring_pairs/numkin_strat_pairs.parquet,
         out/best_numkin.txt

Run:  python nbs/scoring_03a_num_kin.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import scoring_util as su

import kdata
from katlas.scoring import multiply

SEED = 0
VAL_FRAC = 0.2
NUMKINS = [1, 3, 5, 10, 20, 40, 80, 120, 160, 200, 'all']
BINS, LABELS = [0, 1, 3, 10, 40, 10 ** 9], ['1', '2-3', '4-10', '11-40', '>40']


def build_branches(split, pools, common, cols):
    """Per-branch features: inner-train (fits), held-out VALIDATION (selects the cutoff), TEST
    (generalization). The val carve is leak-free (whole backbones), mirroring scoring_02a."""
    tr_split = split[~split[f'test_{SEED}']]                          # test held out entirely
    vmask = su.strat_group_split(tr_split, frac=VAL_FRAC, seed=1000 + SEED)   # leak-free val carve
    inner, valid = tr_split[~vmask], tr_split[vmask]
    inner = inner[inner.kinase_protein.isin(common)]
    valid = valid[valid.kinase_protein.isin(common)]
    test = split[split[f'test_{SEED}']]
    test = test[test.kinase_protein.isin(common)]

    B = {}
    for bi, branch in enumerate(['ST', 'Tyr']):
        pool = pools[branch]
        pidx = {k: i for i, k in enumerate(pool)}
        tr = su.branch_split(inner)[bi]
        tr = tr[tr.kinase_protein.isin(pool)]
        va = su.branch_split(valid)[bi]
        va = va[va.kinase_protein.isin(pool)]
        te = su.branch_split(test)[bi]
        B[branch] = dict(pool=pool, tr=tr, nk=tr.num_kin.values,
                         Xtr=su.onehot(tr.site_seq.values, cols),
                         ytr=np.array([pidx[k] for k in tr.kinase_protein]),
                         eval={'val': (va, su.onehot(va.site_seq.values, cols)),
                               'test': (te, su.onehot(te.site_seq.values, cols))})
        print(f'{branch}: inner {len(tr)} | val {len(va)} | test {len(te)} | pool {len(pool)}')
    return B


def training_sweep(B, site_kin):
    "Retrain at each num_kin cutoff on inner-train; score VALIDATION (selects) and TEST (generalization)."
    out = []
    for branch, f in B.items():
        for nk in NUMKINS:
            m = np.ones(len(f['nk']), bool) if nk == 'all' else (f['nk'] <= nk)
            model, _ = su.fit_mlp(f['Xtr'][m], f['ytr'][m], len(f['pool']),
                                  hidden=512, depth=2, seed=SEED,
                                  groups=f['tr'].site_seq.str.upper().to_numpy()[m])
            for sp, (ev, Xev) in f['eval'].items():
                p = su.score_pairs(ev, su.mlp_scores(model, Xev), f['pool'], site_kin,
                                   'CDDM-seq MLP', SEED, branch)
                p['train_numkin'], p['split'] = str(nk), sp
                out.append(p)
            print(f'  {branch} num_kin<={nk}: n_train={int(m.sum())}')
    return pd.concat(out, ignore_index=True)


def test_stratification(B, site_kin, pspa):
    "MLP (num_kin<=40) and PSPA over the full test; bin by test-site num_kin downstream."
    out = []
    for branch, f in B.items():
        pool = f['pool']
        te, Xte = f['eval']['test']                                   # stratification is on TEST (descriptive)
        m40 = f['nk'] <= su.BASE_NUMKIN
        model, _ = su.fit_mlp(f['Xtr'][m40], f['ytr'][m40], len(pool), hidden=512, depth=2, seed=SEED,
                              groups=f['tr'].site_seq.str.upper().to_numpy()[m40])
        out.append(su.score_pairs(te, su.mlp_scores(model, Xte), pool, site_kin,
                                  f'CDDM-seq MLP (nk<={su.BASE_NUMKIN})', SEED, branch))
        out.append(su.score_pairs(te, su.gen_scores(te, pspa.loc[pool], multiply, pool),
                                  pool, site_kin, 'PSPA', SEED, branch))
    return pd.concat(out, ignore_index=True)


def main():
    split = su.load_split()
    pspa = kdata.load('pspa')
    site_kin = su.site_kinase_map(split)
    pools, _ = su.load_pool()
    common = pools['ST'] + pools['Tyr']

    B = build_branches(split, pools, common, su.onehot_cols())

    print('\n== 1. training num_kin sweep ==')
    sweep = training_sweep(B, site_kin)
    sweep.to_parquet(su.RES / 'numkin_sweep_pairs.parquet')
    print('saved', su.RES / 'numkin_sweep_pairs.parquet', sweep.shape, '| splits:', sorted(sweep.split.unique()))

    # cutoff SELECTED on VALIDATION (never test), mirroring the flank-window sweep in scoring_02a
    valm = su.summarize(sweep[sweep.split == 'val'], ['train_numkin']).set_index('train_numkin')['top10']
    testm = su.summarize(sweep[sweep.split == 'test'], ['train_numkin']).set_index('train_numkin')['top10']
    best = valm.idxmax()
    (su.OUT / 'best_numkin.txt').write_text(str(best))
    print(f'selected num_kin cutoff = {best} (VALIDATION argmax micro recall@10) -> out/best_numkin.txt')
    print('  val :', {k: round(v, 3) for k, v in valm.items()})
    print('  test:', {k: round(v, 3) for k, v in testm.items()})

    print('\n== 2. test stratification ==')
    strat = test_stratification(B, site_kin, pspa)
    strat.to_parquet(su.RES / 'numkin_strat_pairs.parquet')
    print('saved', su.RES / 'numkin_strat_pairs.parquet', strat.shape)

    strat = strat.assign(test_num_kin=pd.cut(strat.num_kin, BINS, labels=LABELS))
    print(su.summarize(strat, ['branch', 'test_num_kin', 'method']).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
