"""motif_02 · PSPA derived matrices: per-position scale + per-position log-enrichment.

The `pspa` dataset holds the normalized PSPA (positional scanning peptide array) values.
From it this builds two derived per-kinase PSSMs:

  pspa_scale    each position rescaled so the 20 amino-acid probabilities sum to 1, making
                PSPA directly comparable to the CDDM frequency PSSMs. A reference dataset
                (many consumers load it via `kdata.load('pspa_scale')`), so it stays in the
                katlas_datasets store; the version it replaces is kept in `_archive/`.

  pspa_enrich   per-position median log-enrichment: log2(val / per-position median over the
                position's value-bearing cells), zeros -> NaN (no +inf).
                A signed, per-position-centred matrix (the PSPA analogue of the CDDM log-odds /
                surface-display enrichment). Canonical copy in the kdata store
                (`kdata.load('pspa_enrich')`), mirrored into the flat `pssm/` store for the
                kinase-method table; one row per kinase.

It also copies the raw (pre-normalisation) PSPA (`kdata.load('pspa_raw')`) into `pssm/pspa_raw.parquet`,
so every PSPA view (raw / scale / enrich) is available from the store the web + kinase-method table read.

Both `pspa_scale` and `pspa_enrich` derive independently from the normalized `pspa`; neither depends
on the other.

Inputs   kdata.load('pspa'), kdata.load('pspa_raw')
Outputs  katlas_datasets/PSPA/pspa_all_scale.parquet    (pspa_scale, via kdata)
         katlas_datasets/PSPA/pspa_all_enrich.parquet   (pspa_enrich, via kdata)
         pssm/pspa_enrich.parquet, pssm/pspa_raw.parquet   (flat-store mirrors)

Run:  python nbs/motif_02_pspa_enrich_scale.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from paths import PSSM

import kdata
from katlas.pssm import clean_zero_normalize, flatten_pssm, recover_pssm


def normalize_pspa_row(r):
    "Recover the flat row into a position x aa matrix, normalize each position, reflatten."
    return flatten_pssm(clean_zero_normalize(recover_pssm(r)))


def pspa_enrich(freq):
    "Per-position median log-enrichment of PSPA: log2(val / per-position median); zeros -> NaN."
    out = {}
    for k in freq.index:
        # recover_pssm pads each position to the full residue alphabet, filling absent cells
        # (the 17 non-acceptor residues at position 0, the untested +5 for S/T kinases) with 0.
        # Take the median over the value-bearing cells only (m>0), NOT the padded column: a
        # padded median can be 0 and turn the acceptor cells into +inf. Masking first makes both
        # numerator and denominator ignore the structural zeros, so absent positions stay NaN.
        m = recover_pssm(freq.loc[k]).where(lambda x: x > 0)
        out[k] = flatten_pssm(np.log2(m.div(m.median(axis=0), axis=1)))
    return pd.DataFrame(out).T


def main():
    print('dataset store:', kdata.DATASET)

    pspa_full = kdata.load('pspa')              # keeps +5 (NaN for S/T kinases, which stop at +4)
    pspa = pspa_full.dropna(axis=1)             # column-uniform view (-5..+4) for pspa_scale
    print('pspa:', pspa_full.shape, '| column-uniform for scale:', pspa.shape)

    pspa_scale = pspa.apply(lambda r: pd.Series(normalize_pspa_row(r)), axis=1)
    print('pspa_scale:', pspa_scale.shape)
    kdata.save('pspa_scale', pspa_scale)

    # compute enrich per kinase from the FULL matrix, so tyrosine kinases keep their +5 position
    # (S/T kinases have no +5, so it stays NaN for them)
    enrich = pspa_enrich(pspa_full)
    print('pspa_enrich:', enrich.shape)
    kdata.save('pspa_enrich', enrich)           # canonical -> kdata.load('pspa_enrich')
    PSSM.mkdir(exist_ok=True)
    enrich.to_parquet(PSSM / 'pspa_enrich.parquet')   # mirror for the flat pssm/ method table
    print('wrote', PSSM / 'pspa_enrich.parquet')

    # also materialise the raw (pre-normalisation) PSPA into the pssm/ store, so the web /
    # kinase-method table read every PSPA view from one place
    raw = kdata.load('pspa_raw')
    raw.to_parquet(PSSM / 'pspa_raw.parquet')
    print('wrote', PSSM / 'pspa_raw.parquet', raw.shape)


if __name__ == '__main__':
    main()
