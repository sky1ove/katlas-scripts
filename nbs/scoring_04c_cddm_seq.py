"""scoring_04c · The three discriminative CDDM-seq models, at ±5.

Instead of building a PSSM and scoring with it (generative), train a classifier that maps a
one-hot site sequence straight to a kinase:

  CDDM-seq Linear       plain cross-entropy
  CDDM-seq MLP          512×2, masked-CE + class-balanced CE
  CDDM-seq MLP + PSPA   the same, plus a learned γ·PSPA score channel

All at ±5, the window scoring_02a's validation sweep selects; the generative CDDM picks the same
±5 in scoring_04b / scoring_02b, so one window serves both families. Evaluated over the headline
RepeatedStratifiedGroupKFold (K=5 × R=3): each model is refit on every fold's own train (90 fits =
3 models × 2 branches × 15 folds), and the pairs stamp `seed = repeat`.

**masked-CE** drops a promiscuous site's OTHER true kinases from the softmax denominator, so a
co-true kinase is not penalised as a negative. The mask is built from TRAIN-ONLY co-labels: a
full-dataset mask leaks test co-labels for sequence-shared sites (56% ST / 79% Tyr of train rows)
and inflated these models. After de-leaking, its effect is modest and macro-leaning — the bulk of
the MLP's edge over Linear is capacity, not the loss.

**Class-balanced CE** (weight ∝ 1/√n_k) because plain CE is frequency-dominated and starves the
rare kinases that drive macro: it lifts macro@5 ~+0.04 / macro@10 ~+0.05 for ~−0.01 micro, and the
MLP then beats generative CDDM on macro too. Linear keeps plain CE.

Inputs   out/scoring_split.parquet, out/scoring_pool.parquet, out/scoring_cddm_seed0.parquet (key space), pspa
Outputs  out/scoring_pairs/cddm_seq_pairs.parquet

Run:  python nbs/scoring_04c_cddm_seq.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import scoring_util as su

import kdata
from katlas.scoring import multiply

WINDOW = 5
TRAIN_CAP = 70000

#: (display name, hidden units | None for linear, use PSPA channel, masked-CE, class-balance)
MODELS = [('CDDM-seq Linear', None, False, False, False),
          ('CDDM-seq MLP', 512, False, True, True),
          ('CDDM-seq MLP + PSPA', 512, True, True, True)]


def main():
    split = su.load_split()
    pspa = kdata.load('pspa')
    # FULL labels - eval only. Multi-label AP/MAP must credit every true kinase of a site.
    site_kin = su.site_kinase_map(split)
    pools, _ = su.load_pool()
    common = pools['ST'] + pools['Tyr']
    cols_w = su.onehot_cols(WINDOW)
    print(f'window ±{WINDOW}: {2 * WINDOW + 1}-mer, one-hot D={len(cols_w)} '
          f'(vs {len(su.onehot_cols())} at ±20)')

    res = []
    for rep, fold, train, test in su.iter_folds(split):
        fseed = rep * su.N_FOLDS + fold                    # unique per-fold RNG seed (0..14)
        print(f'repeat {rep} fold {fold}')
        site_kin_train = su.site_kinase_map(train)         # TRAIN-ONLY co-labels for the mask (this fold)
        train_c = train[(train.num_kin <= su.BASE_NUMKIN)
                        & (train.kinase_protein.isin(common))]
        test = test[test.kinase_protein.isin(common)]

        for bi, branch in enumerate(['ST', 'Tyr']):
            pool = pools[branch]
            pidx = {k: i for i, k in enumerate(pool)}
            tr = su.branch_split(train_c)[bi]
            tr = tr[tr.kinase_protein.isin(pool)].sample(n=min(TRAIN_CAP, len(tr)), random_state=fseed)
            te = su.branch_split(test)[bi]

            Xtr = su.onehot(tr.site_seq.values, cols_w)
            ytr = np.array([pidx[k] for k in tr.kinase_protein])
            Xte = su.onehot(te.site_seq.values, cols_w)

            mask = su.otk_mask(tr.site_seq.values, ytr, pool, site_kin_train)
            cbal = su.class_balance(tr.kinase_protein, pool)
            pspa_b = pspa.loc[pool]
            PStr = su.gen_scores(tr, pspa_b, multiply, pool)
            PSte = su.gen_scores(te, pspa_b, multiply, pool)

            for name, hidden, use_pspa, masked, balance in MODELS:
                print(f'  {branch} {name}')
                model, gamma = su.fit_mlp(Xtr, ytr, len(pool), hidden=hidden, depth=2,
                                          pspa=(PStr if use_pspa else None),
                                          mask=(mask if masked else None),
                                          class_weight=(cbal if balance else None), seed=fseed,
                                          groups=tr.site_seq.str.upper().to_numpy())
                sc = su.mlp_scores(model, Xte, gamma=gamma, pspa=(PSte if use_pspa else None))
                res.append(su.score_pairs(te, sc, pool, site_kin, name, rep, branch))

    pairs = pd.concat(res, ignore_index=True)
    out = su.RES / 'cddm_seq_pairs.parquet'
    pairs.to_parquet(out)
    print('saved', out, pairs.shape)
    print(su.summarize(pairs, ['branch', 'method']).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
