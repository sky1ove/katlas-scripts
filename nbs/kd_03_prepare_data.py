"""kd_03 · Pair each kinase domain's features with the specificity to predict.

The modelling question: can a kinase's *domain sequence* predict its *substrate motif*? This builds
the training tables — three targets × four feature representations — one row per kinase. There is no
held-out test split: kd_04b evaluates every cell by repeated subfamily-grouped cross-validation over
each target's *whole* labelled set (see kd_04b), and deployment (kd_07) refits on that same whole set.

**Targets** are flattened PSSMs, one row per kinase:

  pspa       scaled PSPA, already a distribution at each position. The `_TYR` dual-specificity
             entries are dropped (they overlap the main kinase with a noisier label), and the
             *flanking* phospho-serine columns are dropped because the array cannot tell pS from pT —
             it measures pT, so off-centre `{pos}s` merely duplicates the measured `{pos}t`. Keeping
             `t` (not `s`) matches PSPA's measured pT and the cross-method alignment (align on `t`,
             drop `s`); the central acceptor `0s`/`0t`/`0y` are kept, being the genuinely distinct
             acceptor identity.
  mlp_attr   the CDDM MLP-attribution PSSM (motif_17). Count-robust (unlike the CDDM frequency,
             biased by a kinase's site count) and it distinguishes pS from pT, so its s/t stay
             separate. It is **signed** (not a per-position distribution); kept in raw attribution
             units — a per-position rescale would not change the rank-based scoring and would lose
             the depletion (negative) signal.

Inputs   out/kd_feat_{onehot,onehot_pca,t5,esm}.parquet (kd_02a/kd_02b),
         out/mlp_attr_pssm_full.parquet (motif_17), kdata: pspa_scale, cddm, kinase_info
Outputs  out/kd_train_{pspa,mlp_attr,cddm}_{onehot,onehot_pca,esm,t5}.parquet

Run:  python nbs/kd_03_prepare_data.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from paths import OUT

import kdata
from katlas.pssm import flatten_pssm, recover_pssm

FEATURES = ['onehot', 'onehot_pca', 'esm', 't5']


def kinase_to_kd_id():
    "kinase name -> kd_ID, for the non-pseudo kinases."
    info = kdata.load('kinase_info')
    info = info[info.pseudo == '0']
    return info.set_index('kinase')['kd_ID']


def row_flatten(pssms):
    "Re-flatten each row position-major, so it reshapes into position x residue."
    return pssms.apply(lambda r: pd.Series(flatten_pssm(recover_pssm(r), column_wise=False)),
                       axis=1)


def build_target(label, pssms, id_map):
    "One target PSSM table, re-keyed to kd_ID and row-wise flattened."
    if label == 'pspa':
        pssms = pssms[~pssms.index.str.contains('_')]        # drop the _TYR dual-specificity duplicates

    mapped = pssms.index.to_series().map(id_map)
    unmapped = int(mapped.isna().sum())
    if unmapped:
        print(f'  {unmapped} kinases have no kd_ID (pseudokinase or no active domain) - dropped')
    pssms = pssms[mapped.notna().values]
    pssms.index = mapped.dropna()

    out = row_flatten(pssms)
    if label == 'pspa':
        # the array measures pT, so off-centre `{pos}s` merely duplicates `{pos}t`; drop the `s` copy
        # and keep the measured `t`. Central 0s/0t/0y are the distinct acceptor identity — kept.
        dup_s = [c for c in out.columns
                 if (m := re.fullmatch(r'(-?\d+)s', str(c))) and int(m.group(1)) != 0]
        out = out.drop(columns=dup_s)
        print(f'  dropped {len(dup_s)} duplicate flanking pS/pT (s) columns')
    print(f'  target {out.shape}')
    return out


def load_targets():
    "The raw target PSSMs, keyed by kinase name."
    cddm = kdata.load('cddm')
    flank = [c for c in cddm.columns if (m := re.match(r'(-?\d+)', str(c))) and -5 <= int(m.group(1)) <= 5]
    return {'pspa': kdata.load('pspa_scale'),
            'mlp_attr': pd.read_parquet(OUT / 'mlp_attr_pssm_full.parquet'),
            'cddm': cddm[flank]}       # CDDM substrate frequency, ±5 flank — a per-position distribution


def main():
    id_map = kinase_to_kd_id()
    feats = {f: pd.read_parquet(OUT / f'kd_feat_{f}.parquet') for f in FEATURES}
    print('features:', {k: v.shape for k, v in feats.items()})

    for label, pssms in load_targets().items():
        print(f'\n== {label.upper()} ==')
        target = build_target(label, pssms, id_map)

        no_feature = target.index.difference(feats['t5'].index)
        if len(no_feature):
            print(f'  {len(no_feature)} kinases have specificity but no active-domain feature')
        for fname, feat in feats.items():
            combined = target.merge(feat, left_index=True, right_index=True)
            out = OUT / f'kd_train_{label}_{fname}.parquet'
            combined.to_parquet(out)
            print(f'  {fname:11} {combined.shape} -> {out.name}')


if __name__ == '__main__':
    main()
