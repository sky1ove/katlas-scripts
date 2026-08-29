"""Reference-dataset access for the nbs scripts.

The store lives at `nbs/katlas_datasets/`, right next to the scripts, and every dataset has
one fixed filename - no `_YYYYMMDD` suffixes and no version resolution. A script that
regenerates a dataset overwrites it in place, so every later script reads the new version
with no path to update.

    import kdata
    cddm = kdata.load('cddm')
    ks   = kdata.ks_dataset(thr=40)
    kdata.save('cddm', df)          # archives the old file, then overwrites

`save()` moves the file it is about to replace into `katlas_datasets/_archive/`, keeping the
most recent previous version of each dataset as a one-step undo.

The `katlas` package reads the same folder through `Data.*` (via `KATLAS_DATA_DIR` or its
repo fallback), so the two views never diverge. Scripts use `kdata` because it is explicit
about where the bytes come from; `Data` remains the public API.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATASET = Path(__file__).resolve().parent / 'katlas_datasets'
ARCHIVE = DATASET / '_archive'

#: dataset name -> path relative to DATASET. Fixed names, no date suffixes.
PATHS = {
    # kinase metadata
    'kinase_info':      'kinase_info.csv',
    'kinase_uniprot':   'uniprot_human_keyword_kinase.parquet',
    'kd_uniprot':       'uniprot_kd_labeled.parquet',

    # PSPA (positional scanning peptide array)
    'pspa':             'PSPA/pspa_all_norm.parquet',
    'pspa_raw':         'PSPA/pspa_all_raw.parquet',
    'pspa_scale':       'PSPA/pspa_all_scale.parquet',
    'pspa_enrich':      'PSPA/pspa_all_enrich.parquet',
    'pspa_st':          'PSPA/pspa_st_norm.parquet',
    'pspa_tyr':         'PSPA/pspa_tyr_norm.parquet',
    'pspa_st_pct':      'PSPA/pspa_pct_st.parquet',
    'pspa_tyr_pct':     'PSPA/pspa_pct_tyr.parquet',
    'pspa_num':         'PSPA/pspa_divide_num.csv',

    # kinase-substrate dataset + CDDM
    'ks_dataset':       'CDDM/ks_datasets.parquet',
    'ks_unique':        'CDDM/unique_ks_sites.parquet',
    'ks_background':    'CDDM/ks_background.parquet',
    'cddm':             'CDDM/pssms.parquet',
    'cddm_upper':       'CDDM/pssms_upper.parquet',
    # log-odds against the ks_dataset background (built by motif_01)
    'cddm_LO':          'CDDM/pssms_LO.parquet',
    'cddm_LO_upper':    'CDDM/pssms_LO_upper.parquet',

    # phosphosites
    'psp_human_site':             'phosphosites/psp_human.parquet',
    'ochoa_site':                 'phosphosites/ochoa_site.parquet',
    'combine_site_psp_ochoa':     'phosphosites/combine_site_psp_ochoa.parquet',
    'combine_site_phosphorylated': 'phosphosites/phosphorylated_combine_site.parquet',
    'human_site':                 'phosphosites/phosphorylated_combine_site20.parquet',
    'cptac_ensembl_site':         'phosphosites/linkedOmicsKB_ref_pan.parquet',
    'cptac_gene_site':            'phosphosites/linkedOmics_ref_pan.parquet',
    'cptac_unique_site':          'phosphosites/cptac_unique_site.parquet',

    # amino acids / pathways
    'aa_info':            'amino_acids/aa_info.parquet',
    'aa_rdkit':           'amino_acids/aa_rdkit.parquet',
    'aa_morgan':          'amino_acids/aa_morgan.parquet',
    'reactome_pathway':    'reactome_all_levels.parquet',
    'reactome_pathway_lo': 'reactome_lowest_level.parquet',
}


def path(name: str) -> Path:
    "Absolute path of a dataset. Raises KeyError for an unknown name."
    try:
        return DATASET / PATHS[name]
    except KeyError:
        raise KeyError(f'unknown dataset {name!r}; known: {sorted(PATHS)}') from None


@lru_cache
def _read(p: str) -> pd.DataFrame:
    return pd.read_csv(p) if p.endswith('.csv') else pd.read_parquet(p)


def load(name: str) -> pd.DataFrame:
    "Read a dataset by name. Cached; call `clear_cache()` after overwriting one in the same process."
    p = path(name)
    if not p.exists():
        raise FileNotFoundError(f'{name} not found at {p}')
    return _read(str(p)).copy()


def clear_cache() -> None:
    "Forget cached reads, so a dataset rewritten in this process is re-read from disk."
    _read.cache_clear()


def save(name: str, df: pd.DataFrame, index: bool | None = None) -> Path:
    "Overwrite a dataset, moving the version it replaces into `katlas_datasets/_archive/`."
    p = path(name)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists():
        old = ARCHIVE / PATHS[name]
        old.parent.mkdir(parents=True, exist_ok=True)
        p.replace(old)
        print(f'  archived previous {name} -> {old}')

    if p.suffix == '.csv':
        df.to_csv(p, index=False if index is None else index)
    else:
        df.to_parquet(p, index=index)

    clear_cache()
    print(f'  wrote {name} -> {p} {df.shape}')
    return p


# ---------------------------------------------------------------- derived views


def ks_dataset(thr: int | None = 40) -> pd.DataFrame:
    "Kinase-substrate pairs, keeping only sites targeted by <= `thr` kinases (None = no filter)."
    df = load('ks_dataset')
    return df if thr is None else df[df['num_kin'] <= thr].copy()


if __name__ == '__main__':
    print('DATASET:', DATASET)
    for n in sorted(PATHS):
        p = path(n)
        print(f'  {"ok " if p.exists() else "MISS"} {n:28} {PATHS[n]}')
