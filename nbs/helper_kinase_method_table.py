"""helper · Per-kinase method-coverage table — which specificity matrices exist for each kinase.

One row per kinase, one boolean column per method matrix (does that kinase have that data?),
plus `n_methods` and the kinase_info metadata (gene/uniprot/group/family). Covers the CDDM,
PSPA and surface-display matrices in the `pssm/` store + kdata, and the scoring MLP-attribution
matrix if it has been built. The index is every kinase that has at least one method.

Method columns come from these sources:
  cddm_freq     kdata `cddm`            cddm_logodds  kdata `cddm_LO`
  pspa          kdata `pspa` (norm)     pspa_raw      pssm/pspa_raw       pspa_enrich  pssm/pspa_enrich
  sd_p1_freq    pssm/sd_freq_p1         sd_p1_enrich  pssm/sd_ptyr_p1
  sd_p2_freq    pssm/sd_freq_p2         sd_p2_enrich  pssm/sd_ptyrvar_p2  sd_p2_x5yx5  pssm/sd_x5yx5_p2
  mlp_attr      out/mlp_attr_pssm_full.parquet (motif_17; skipped if absent)

Inputs   kdata: cddm, cddm_LO, pspa, kinase_info; pssm/*.parquet; out/mlp_attr_pssm_full.parquet
Outputs  out/kinase_method_table.parquet, out/kinase_method_table.csv

Run:  python nbs/helper_kinase_method_table.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from paths import OUT, PSSM

import kdata

#: method column -> a callable returning the matrix whose index is the covered kinases.
#: order here is the column order in the output table.
METHODS = {
    'cddm_freq':    lambda: kdata.load('cddm'),
    'cddm_logodds': lambda: kdata.load('cddm_LO'),
    'pspa':         lambda: kdata.load('pspa'),
    'pspa_raw':     lambda: pd.read_parquet(PSSM / 'pspa_raw.parquet'),
    'pspa_enrich':  lambda: pd.read_parquet(PSSM / 'pspa_enrich.parquet'),
    'sd_p1_freq':   lambda: pd.read_parquet(PSSM / 'sd_freq_p1.parquet'),
    'sd_p1_enrich': lambda: pd.read_parquet(PSSM / 'sd_ptyr_p1.parquet'),
    'sd_p2_freq':   lambda: pd.read_parquet(PSSM / 'sd_freq_p2.parquet'),
    'sd_p2_enrich': lambda: pd.read_parquet(PSSM / 'sd_ptyrvar_p2.parquet'),
    'sd_p2_x5yx5':  lambda: pd.read_parquet(PSSM / 'sd_x5yx5_p2.parquet'),
    'mlp_attr':     lambda: pd.read_parquet(OUT / 'mlp_attr_pssm_full.parquet'),
}

META_COLS = ['gene', 'uniprot', 'group', 'family']


def method_kinases():
    "Map each method column to the set of kinases it covers; skip a source that isn't built yet."
    out = {}
    for name, loader in METHODS.items():
        try:
            out[name] = set(loader().index)
            print(f'  {name:13} {len(out[name])} kinases')
        except FileNotFoundError:
            print(f'  skip {name}: source not found')
    return out


def main():
    cov = method_kinases()
    kinases = sorted(set().union(*cov.values()))
    print(f'{len(kinases)} kinases with >= 1 method')

    table = pd.DataFrame(index=pd.Index(kinases, name='kinase'))
    for name in METHODS:                                   # keep declared column order; absent source -> all False
        table[name] = table.index.isin(cov.get(name, set()))
    table.insert(0, 'n_methods', table[list(METHODS)].sum(axis=1))

    info = kdata.load('kinase_info').drop_duplicates('kinase').set_index('kinase')
    meta = info.reindex(table.index)[META_COLS]
    table = pd.concat([meta, table], axis=1)

    OUT.mkdir(exist_ok=True)
    table.to_parquet(OUT / 'kinase_method_table.parquet')
    table.to_csv(OUT / 'kinase_method_table.csv')
    print('wrote', OUT / 'kinase_method_table.parquet', '+ .csv', table.shape)
    print('per-method totals:\n', table[list(cov)].sum().to_string())


if __name__ == '__main__':
    main()
