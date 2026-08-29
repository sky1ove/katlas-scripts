"""motif_10 · Build the surface-display PSSMs.

Two bacterial peptide-display screens of tyrosine-kinase specificity:

  paper1  eLife 35190 (Shah/Kuriyan)  Human-pTyr library, 15-aa peptides (+-7), central Y.
          SRC, LCK, ABL1, ZAP70.                                raw/sd_paper1_enrich.csv
  paper2  eLife 82345 (Li/Shah)       pTyr-Var library, 11-aa peptides (+-5), central Y.
          SRC, FYN, HCK, ABL1, JAK2, FER, FES, FGFR1, FGFR3,
          EPHB1, EPHB2, MERTK.                                  raw/sd_paper2_enrich.csv

The CSVs hold per-peptide enrichment `E_p = f_sorted / f_input` - the maximal raw data the
papers released (read counts are not public, so their exact read-weighted figures cannot be
recomputed). From `E_p` this builds two PSSM families:

  enrichment (`ptyr_p1`, `ptyrvar_p2`)
      mean(log2 E) over all E>0 peptides -> mask cells with n <= MIN_COUNT -> per-position
      median centering. mean-of-log2 is the outlier-robust geometric mean, strictly better
      than the papers' log2(mean E), which a few extreme enrichments dominate (the -5P /
      -7P / +3W artefacts). The n<=3 mask blanks cells backed by three or fewer peptides,
      which are unreliable estimates. No E cutoff, matching the papers' enrichment heatmaps.

  frequency (`freq_p1`, `freq_p2`)
      composition of the high-efficiency foreground E>=1.5, like the papers' pLogo / Fig 5A.
      No count mask - a low count here is a genuinely low frequency. paper2 uses reference
      (non-variant) peptides only.

`x5yx5_p2` is the *published* paper2 X5-Y-X5 matrix (a random library; no per-peptide data
to recompute), passed through unchanged for comparison.

Per-kinase logo heatmaps are rendered by motif_11b (one kinase) and web_01c (batch) from
`pssm/sd_*.parquet`, so this script only writes the flat parquets. Method notes: METHODS.md.

Inputs   raw/sd_paper1_enrich.csv, raw/sd_paper2_enrich.csv,
         raw/sd_paper2_fig2data1_X5YX5_matrices.xlsx
Outputs  pssm/sd_{ptyr_p1,ptyrvar_p2,x5yx5_p2,freq_p1,freq_p2}.parquet

Run:  python nbs/motif_10_sd_build_pssm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from paths import PSSM, RAW

from katlas.pssm import flatten_pssm, get_prob

#: 20 standard residues - display peptides carry no phospho-priming
AA20 = list('PGACSTVILMFYWHKRQNDE')

PAPER1_KIN = ['SRC', 'LCK', 'ABL1', 'ZAP70']
PAPER2_KIN = ['SRC', 'FYN', 'HCK', 'ABL1', 'JAK2', 'FER', 'FES',
              'FGFR1', 'FGFR3', 'EPHB1', 'EPHB2', 'MERTK']

MIN_COUNT = 3       # enrichment only: mask cells backed by this many peptides or fewer
FREQ_THR = 1.5      # frequency only: the papers' high-efficiency foreground cutoff

X5_SHEETS = {'Src 1Y': 'SRC', 'Abl 1Y': 'ABL1', 'Fer 1Y': 'FER',
             'EPHB1 1Y': 'EPHB1', 'EPHB2 1Y': 'EPHB2'}


# ---------------------------------------------------------------- loading


def load_enrich(path):
    "Read an enrichment CSV, stripping the BOM/whitespace from column names."
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip('﻿') for c in df.columns]
    return df


def load_papers():
    "Both enrichment tables, restricted to the peptides each paper's analysis is eligible for."
    p1 = load_enrich(RAW / 'sd_paper1_enrich.csv')          # already WT phosphosites, all central Y
    p2 = load_enrich(RAW / 'sd_paper2_enrich.csv')

    # paper2 eligibility (reproduces their 7382): central Y and exactly one Y in the peptide
    p2 = p2[(p2.seq.str[5] == 'Y') & (pd.to_numeric(p2['# of Ys'], errors='coerce') == 1)].copy()
    p2['is_ref'] = p2['name'] == (p2['Gene'].astype(str) + '_' + p2['Phosphosite'].astype(str))

    print(f'paper1: {len(p1)} peptides')
    print(f'paper2 eligible: {len(p2)} | reference: {int(p2.is_ref.sum())} '
          f'| variant: {int((~p2.is_ref).sum())}')
    return p1, p2


def load_sheet(path, sheet):
    "Load a published matrix sheet (aa x position) from a paper2 source-data workbook."
    d = pd.read_excel(path, sheet_name=sheet, header=None).set_index(0)
    d.columns = [int(x) for x in d.iloc[0]]
    return d.iloc[1:].astype(float)


def peptide_positions(seqs):
    "Position labels for a fixed-width peptide library, centred on the acceptor."
    L = len(seqs[0])
    ci = L // 2
    return list(range(-ci, L - ci)), np.array([list(s) for s in seqs])


# ---------------------------------------------------------------- PSSM builders


def geomean_pssm(df, k):
    "Enrichment PSSM: mean(log2 E) over E>0 peptides -> n<=MIN_COUNT mask -> per-position median centering."
    E = pd.to_numeric(df[k], errors='coerce').to_numpy()
    L = np.where(E > 0, np.log2(np.where(E > 0, E, 1.0)), np.nan)
    positions, arr = peptide_positions(df.seq.to_numpy())

    M = pd.DataFrame(index=AA20, columns=positions, dtype=float)
    C = pd.DataFrame(0, index=AA20, columns=positions, dtype=int)
    for jj, pos in enumerate(positions):
        col = arr[:, jj]
        for a in AA20:
            m = (col == a) & np.isfinite(L)
            C.loc[a, pos] = int(m.sum())
            M.loc[a, pos] = L[m].mean() if m.sum() else np.nan

    M = M.mask(C <= MIN_COUNT)
    M = M.sub(M.median(axis=0), axis=1)
    return M, int(np.isfinite(L).sum())


def freq_pssm(df, k, ref_only=False):
    "Frequency PSSM over the foreground E >= FREQ_THR; ref_only drops the variant peptides."
    d = df.assign(_e=pd.to_numeric(df[k], errors='coerce'))
    d = d[d['_e'] >= FREQ_THR]
    if ref_only and 'is_ref' in d.columns:
        d = d[d.is_ref]
    return get_prob(d['seq']).reindex(AA20), len(d)


def build_all(p1, p2):
    "The five surface-display PSSM sets, each as {kinase: matrix} plus the peptide counts."
    specs = {}

    for name, df, kins in [('ptyr_p1', p1, PAPER1_KIN), ('ptyrvar_p2', p2, PAPER2_KIN)]:
        built = {k: geomean_pssm(df, k) for k in kins}
        specs[name] = ({k: m for k, (m, _) in built.items()},
                       {k: n for k, (_, n) in built.items()})

    for name, df, kins, ref_only in [('freq_p1', p1, PAPER1_KIN, False),
                                     ('freq_p2', p2, PAPER2_KIN, True)]:
        built = {k: freq_pssm(df, k, ref_only) for k in kins}
        specs[name] = ({k: m for k, (m, _) in built.items()},
                       {k: n for k, (_, n) in built.items()})

    x5 = RAW / 'sd_paper2_fig2data1_X5YX5_matrices.xlsx'
    mats = {gene: load_sheet(x5, sheet).reindex(AA20) for sheet, gene in X5_SHEETS.items()}
    specs['x5yx5_p2'] = (mats, {k: None for k in mats})      # published matrix, no per-peptide n
    return specs


def main():
    p1, p2 = load_papers()
    specs = build_all(p1, p2)

    for name, (mats, ns) in specs.items():
        flat = pd.DataFrame({k: flatten_pssm(m) for k, m in mats.items()}).T
        out = PSSM / f'sd_{name}.parquet'   # write straight into the PSSM store the plotters read
        flat.to_parquet(out)
        counted = {k: v for k, v in ns.items() if v is not None}
        n_note = f' | peptides {min(counted.values()):,}-{max(counted.values()):,}' if counted else ''
        print(f'{name:12} {flat.shape[0]} kinases -> {out}{n_note}')


if __name__ == '__main__':
    main()
