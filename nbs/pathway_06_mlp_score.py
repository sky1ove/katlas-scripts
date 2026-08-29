"""pathway_06 · Score the human phosphoproteome with the full-data CDDM-seq MLP.

The third scoring method for the pathway analysis, alongside PSPA and CDDM. Where pathway_01 scores
sites with the generative PSSMs (CDDM, PSPA), this scores them with the discriminative MLP - the
same full-data model motif_17 distils into the mlp_attr PSSM, but used directly as a scorer rather
than as an attribution matrix.

Trains one MLP per branch (ST, Tyr) on all CDDM sites (config matches motif_17 / scoring_04c:
one-hot ±5, masked-CE, class-balance, 512x2), then scores every unique phosphosite. Each site gets
a logit for every kinase in its branch's pool; the two branches are concatenated so the output has
the same shape/behaviour as pathway_cddm_score (a Ser site scores low for a Tyr kinase and vice
versa, since position 0 is in the one-hot).

Output matches pathway_01's score files exactly (site_seq, sub_site, sub_gene, acceptor + one column
per kinase), so pathway_01b (site-centric) and a top-2% selection run on it unchanged.

Inputs   out/pathway_cddm_score.parquet (for the identical site set + sub_site/sub_gene/acceptor),
         kdata: ks_dataset, kinase_info
Outputs  out/pathway_mlp_score.parquet   scored sites (one row per unique site_seq)

Run:  python nbs/pathway_06_mlp_score.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import scoring_util as su
from paths import OUT

import kdata

WINDOW = 5
SEED = 0
MIN_SITES = 20      # full-data count floor for the deployment pool (matches motif_17)
MIN_TRAIN = 5       # a kinase needs at least this many training sites to enter the pool


def load_deployment_set():
    ks = kdata.ks_dataset(thr=None)
    ks['kinase_id'] = ks.kinase_uniprot + '_' + ks.kinase_protein.str.split().str[0]
    info = kdata.load('kinase_info'); info = info[info.pseudo == '0']
    ks = ks[ks.kinase_id.isin(set(info.uniprot + '_' + info.kinase))].copy()
    vc = ks.kinase_id.value_counts()
    ks = ks[ks.kinase_id.isin(set(vc[vc >= MIN_SITES].index))].copy()
    print('deployment set:', ks.shape, '| kinases:', ks.kinase_id.nunique())
    return ks


def train_branch(ks, branch, cols):
    "Train the full-data MLP for one branch; return (model, pool of kinase names)."
    bi = 0 if branch == 'ST' else 1
    gmap = ks.drop_duplicates('kinase_id').set_index('kinase_id').kinase_group
    id2name = ks.drop_duplicates('kinase_id').set_index('kinase_id').kinase_protein
    site_kin_id = ks.groupby('site_seq').kinase_id.agg(set).to_dict()

    db = su.branch_split(ks)[bi]
    trall = db[db.num_kin <= su.BASE_NUMKIN]
    vc = trall.kinase_id.value_counts()
    pool = [k for k in ks.kinase_id.unique()
            if (gmap.get(k) == 'TK') == (branch == 'Tyr') and vc.get(k, 0) >= MIN_TRAIN]
    pidx = {k: i for i, k in enumerate(pool)}
    tr = trall[trall.kinase_id.isin(pool)]

    ytr = np.array([pidx[k] for k in tr.kinase_id])
    Xtr = su.onehot(tr.site_seq.values, cols)
    mask = su.otk_mask(tr.site_seq.values, ytr, pool, site_kin_id)
    cbal = su.class_balance(tr.kinase_id, pool)
    model, _ = su.fit_mlp(Xtr, ytr, len(pool), hidden=512, depth=2, seed=SEED,
                          mask=mask, class_weight=cbal,
                          groups=tr.site_seq.str.upper().to_numpy())
    model.eval()
    print(f'  {branch}: pool={len(pool)} train_sites={len(tr)}')
    return model, [id2name[k] for k in pool]


def main():
    cols = su.onehot_cols(WINDOW)
    base = pd.read_parquet(OUT / 'pathway_cddm_score.parquet')[
        ['site_seq', 'sub_site', 'sub_gene', 'acceptor']].copy()
    print(f'scoring {len(base):,} unique sites with the full-data MLP')

    X = su.onehot(base.site_seq.values, cols)
    ks = load_deployment_set()

    frames = [base]
    for branch in ['ST', 'Tyr']:
        model, pool = train_branch(ks, branch, cols)
        S = su.mlp_scores(model, X)                        # [n_sites, len(pool)]
        df = pd.DataFrame(S, columns=pool, index=base.index)
        df = df.loc[:, ~df.columns.duplicated()]          # a kinase_protein can repeat; keep first
        frames.append(df)

    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]           # guard cross-branch name clashes
    n_kin = out.shape[1] - 4
    out.to_parquet(OUT / 'pathway_mlp_score.parquet')
    print(f'MLP score matrix {out.shape} ({n_kin} kinases) -> {OUT / "pathway_mlp_score.parquet"}')

    # top-2% selection on the MLP scores, for the pathway_07 method comparison. The MLP has no PSSM
    # acceptor distribution, so borrow each kinase's S/T/Y preference from CDDM (same kinase names).
    import pathway_01_score_sites as p1
    cddm = kdata.load('cddm')
    shared = [k for k in out.columns[4:] if k in cddm.index]
    info = p1.build_info(cddm.loc[shared, ['0s', '0t', '0y']], out)
    info.to_parquet(OUT / 'pathway_mlp_info.parquet')
    print(f'MLP top-2% info {info.shape} -> {OUT / "pathway_mlp_info.parquet"}')


if __name__ == '__main__':
    main()
