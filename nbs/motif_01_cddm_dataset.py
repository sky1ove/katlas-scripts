"""motif_01 · Build the CDDM PSSMs from the KS dataset.

CDDM = the per-kinase position-specific scoring matrix computed directly from the
observed substrate site sequences.

Pipeline
  1. `kdata.ks_dataset(thr=40)` - sites targeted by <= 40 kinases, matching the
     `num_kin <= 40` training filter used by the scoring benchmark.
  2. Drop site-sequence duplicates per kinase, compared case-insensitively (sequences
     that differ only in phospho-priming case count once).
  3. Keep kinases with >= 40 remaining substrate sites, so each PSSM is reliable.
  4. Two variants: case-preserving (lowercase s/t/y mark phospho-priming residues) and
     all-uppercase (aligned to the same columns, the s/t/y columns become zero).

Overwrites `cddm` / `cddm_upper`, then derives and saves `cddm_LO` / `cddm_LO_upper` from them and
the `ks_background` (`cddm_log_odds` below: log2(freq) − log2(bg)). This is the **single producer**
of the log-odds, so `motif_00` (which builds `ks_background`) must run first. The versions replaced
are kept in `katlas_datasets/_archive/`.

Inputs   kdata.ks_dataset(thr=40), kdata.load('ks_background')  (ks_background ← motif_00)
Outputs  katlas_datasets/CDDM/pssms.parquet, pssms_upper.parquet,
                              pssms_LO.parquet, pssms_LO_upper.parquet

Run:  python nbs/motif_01_cddm_dataset.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

import kdata
from katlas.lo import get_pssm_LO_flat
from katlas.pssm import flatten_pssm, get_prob, recover_pssm

MIN_SITES = 40


def build_pssms(subset, seq_col):
    "Flat per-kinase PSSM from `seq_col`, one row per kinase."
    return (subset
            .groupby('kinase_protein')
            .apply(lambda g: pd.Series(flatten_pssm(get_prob(g, seq_col))),
                   include_groups=False))


def cddm_log_odds(freq, bg_row):
    "Per-kinase log2(freq) - log2(ks background); NaN where freq or bg is 0 (undefined odds)."
    bg = recover_pssm(kdata.load('ks_background').loc[bg_row])   # bg_row built by motif_00
    return pd.DataFrame([flatten_pssm(get_pssm_LO_flat(flat, bg_pssm=bg))
                         for _, flat in freq.iterrows()], index=freq.index)


def main():
    print('dataset store:', kdata.DATASET)

    data = kdata.ks_dataset(thr=40)
    print('ks_dataset(thr=40):', data.shape)

    data = data.assign(seq_upper=data['site_seq'].str.upper())
    data = data.drop_duplicates(['kinase_protein', 'seq_upper'])
    print('after case-insensitive dedup:', data.shape)

    cnt = data['kinase_protein'].value_counts()
    keep = cnt[cnt >= MIN_SITES].index
    subset = data[data.kinase_protein.isin(keep)]
    print(f'kinases with >= {MIN_SITES} sites: {len(keep)} | rows: {len(subset)}')

    df = build_pssms(subset, 'site_seq')
    print('case-preserving CDDM:', df.shape)

    df_upper = build_pssms(subset, 'seq_upper').reindex(columns=df.columns, fill_value=0.0)
    print('uppercase CDDM      :', df_upper.shape)

    kdata.save('cddm', df)
    kdata.save('cddm_upper', df_upper)

    # log-odds = log2(freq) - log2(ks background); this is their single producer (needs the
    # ks_background from motif_00). NaN where freq or bg is 0 — renders blank, skipped by scoring.
    for name, freq, bg_row in [('cddm_LO', df, 'ks_STY'), ('cddm_LO_upper', df_upper, 'ks_STY_upper')]:
        LO = cddm_log_odds(freq, bg_row)
        print(f'  {name} from {bg_row}: {LO.shape}, NaN cells {int(LO.isna().values.sum())}')
        kdata.save(name, LO)


if __name__ == '__main__':
    main()
