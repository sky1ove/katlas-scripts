"""motif_17 · Distil the CDDM-seq MLP into a per-kinase attribution PSSM.

The discriminative MLP outperforms the generative PSSMs but is opaque: it is a classifier, not a
motif. This turns it back into one, so it can be read and compared like any other PSSM.

**Method: in-silico saturation mutagenesis.** Over a fixed background of real sites, substitute
each (position, residue) in turn and take the mean logit for every kinase, then centre per
(kinase, position). The result is a signed matrix — enriched above zero, depleted below — in the
same `{pos}{aa}` key space as the other PSSMs.

Config matches the shipped model in scoring_04c (±5, masked-CE + class-balance, 512×2,
num_kin ≤ 40) for consistency, not for gain: it is neutral for PSPA recovery (per-kinase Spearman
~0.25, unchanged vs the older ±7 plain-CE), because the recovered motif shape is data-limited,
not loss-limited. Class-balance helps *ranking*, not the per-kinase motif.

The deployment pool is wider than the benchmark pool: full-data `kinase_id` count ≥ 20, non-pseudo,
all CDDM kinases. The per-seed evaluation pool would undercount, and there is no held-out set to
protect here — the attribution is a property of the model, not a prediction.

Also emits a tidy per-(kinase, position, residue) frame of shared attr/PSPA cells (was consumed by
motif_18, now archived in raw_scripts/; kept for the record).
Flanks only; the centre S/T preference is handled there, because raw PSPA's centre is
MAX-normalised and degenerate for a cell-level correlation.

Inputs   kdata: ks_dataset(thr=None), kinase_info, pspa; out/scoring_cddm_seed0.parquet (key space)
Outputs  out/mlp_attr_pssm_full.parquet, out/attr_vs_pspa_aligned.parquet

Run:  python nbs/motif_17_mlp_attr_pssm.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import scoring_util as su
import torch

import kdata

MIN_SITES = 20      # full-data count floor for the deployment pool (the benchmark pool uses 40)
WINDOW = 5
SEED = 0
BG_SITES = 256      # background sites the mutagenesis averages over
MIN_TRAIN = 5       # a kinase needs at least this many training sites to enter the pool


def saturation_attr(model, bg_X, cols, pos_cols, cidx):
    "In-silico saturation mutagenesis -> attribution [K, len(cols)], centred per (kinase, position)."
    M = np.zeros((model[-1].out_features, len(cols)), np.float32)
    with torch.no_grad():
        for c in cols:
            p = int(c[:-1])
            Xc = bg_X.clone()
            Xc[:, pos_cols[p]] = 0        # clear this position, then set the substituted residue
            Xc[:, cidx[c]] = 1
            M[:, cidx[c]] = model(Xc).mean(0).numpy()
    A = M.copy()
    for p in pos_cols:
        A[:, pos_cols[p]] = M[:, pos_cols[p]] - M[:, pos_cols[p]].mean(1, keepdims=True)
    return A


def load_deployment_set():
    "All sites for non-pseudo kinases with at least MIN_SITES of them."
    ks = kdata.ks_dataset(thr=None)
    ks['kinase_id'] = ks.kinase_uniprot + '_' + ks.kinase_protein.str.split().str[0]
    info = kdata.load('kinase_info')
    info = info[info.pseudo == '0']
    ks = ks[ks.kinase_id.isin(set(info.uniprot + '_' + info.kinase))].copy()
    vc = ks.kinase_id.value_counts()
    ks = ks[ks.kinase_id.isin(set(vc[vc >= MIN_SITES].index))].copy()
    print('deployment set:', ks.shape, '| kinases:', ks.kinase_id.nunique())
    return ks


def build_attr(ks, cols):
    cidx = {c: i for i, c in enumerate(cols)}
    pos_cols = {p: [cidx[c] for c in cols if int(c[:-1]) == p]
                for p in sorted({int(c[:-1]) for c in cols})}

    id2name = ks.drop_duplicates('kinase_id').set_index('kinase_id').kinase_protein
    gmap = ks.drop_duplicates('kinase_id').set_index('kinase_id').kinase_group
    site_kin_id = ks.groupby('site_seq').kinase_id.agg(set).to_dict()   # co-labels for masked-CE

    rows = {}
    for bi, branch in enumerate(['ST', 'Tyr']):
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

        bg = tr.sample(n=min(BG_SITES, len(tr)), random_state=1)
        A = saturation_attr(model, torch.tensor(su.onehot(bg.site_seq.values, cols)),
                            cols, pos_cols, cidx)
        for k, i in pidx.items():
            rows[id2name[k]] = A[i]
        print(f'  {branch}: pool={len(pool)} train_sites={len(tr)}')

    attr = pd.DataFrame.from_dict(rows, orient='index', columns=cols)
    return attr[~attr.index.duplicated(keep='first')], gmap


def align_to_pspa(attr, ks, gmap, cols):
    "Tidy per-(kinase, position, residue) frame of the cells attr and PSPA share."
    pspa = kdata.load('pspa')
    pcols = [c for c in cols if c in pspa.columns]
    kinases = [k for k in attr.index if k in pspa.index]
    k2group = ks.drop_duplicates('kinase_protein').set_index('kinase_protein').kinase_group

    long = [(k, k2group.get(k), int(c[:-1]), c[-1], float(attr.loc[k, c]), float(pspa.loc[k, c]))
            for k in kinases for c in pcols if pd.notna(pspa.loc[k, c])]
    aligned = pd.DataFrame(long, columns=['kinase', 'group', 'pos', 'aa', 'attr', 'pspa'])
    print(f'aligned {len(kinases)} kinases × {len(pcols)} shared cells -> {len(aligned)} rows')
    return aligned


def main():
    t0 = time.time()
    cols = su.onehot_cols(WINDOW)
    ks = load_deployment_set()

    attr, gmap = build_attr(ks, cols)
    out = su.OUT / 'mlp_attr_pssm_full.parquet'
    attr.to_parquet(out)
    print(f'attribution PSSM {attr.shape} -> {out} | {time.time() - t0:.0f}s')

    aligned = align_to_pspa(attr, ks, gmap, cols)
    aligned.to_parquet(su.OUT / 'attr_vs_pspa_aligned.parquet')
    print('wrote', su.OUT / 'attr_vs_pspa_aligned.parquet')


if __name__ == '__main__':
    main()
