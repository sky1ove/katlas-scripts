"""helper · Export the supplementary tables.

Writes into `supplement_table/`:

  kinase_info.csv    the 523 kinome-tree kinases and their annotation
  ks_dataset.csv     every kinase-substrate-site pair with its sources
  cddm.xlsx          two sheets - `CDDM` frequency PSSMs and `CDDM_log_odds`

The two big tables are CSV, not Excel, because Excel truncates any cell over 32,767
characters and both carry full protein sequences that exceed it (`human_uniprot_sequence`
reaches 34,350 aa for TTN; `substrate_sequence` / `substrate_phosphoseq` sit at the cap).
CSV keeps every column intact. `cddm` is all numeric, so nothing is lost there and Excel
buys the two-sheets-in-one-file layout - the script still checks the cell limit before
writing and refuses if anything would be truncated.

The log-odds sheet is `kdata.load('cddm_LO')` (built by motif_01): log2(freq) - log2(bg)
against the ks_dataset background (motif_00), which pools all KS-dataset S/T/Y sites deduplicated
on the all-uppercase site sequence.

Inputs   kdata: kinase_info, ks_dataset, cddm, cddm_LO
Outputs  supplement_table/kinase_info.csv, ks_dataset.csv, cddm.xlsx

Run:  python nbs/helper_supplement_tables.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

import kdata

SUPP = Path(__file__).resolve().parent / 'supplement_table'

EXCEL_CELL_LIMIT = 32_767


def max_cell_len(df):
    "Longest string representation across every cell, and the column it is in."
    worst, worst_col = 0, None
    for c in df.columns:
        try:
            m = int(df[c].astype(str).str.len().max())
        except Exception:
            continue
        if m > worst:
            worst, worst_col = m, c
    return worst, worst_col


def write_csv(df, fname):
    out = SUPP / fname
    df.to_csv(out, index=False)
    print(f'  wrote {out} {df.shape} ({out.stat().st_size / 1e6:.1f} MB)')


def write_excel(sheets, fname):
    "Write {sheet_name: df} to one .xlsx, refusing if any cell would be truncated."
    for sheet, df in sheets.items():
        n, col = max_cell_len(df)
        if n > EXCEL_CELL_LIMIT:
            sys.exit(f'{sheet}.{col} has cells up to {n:,} chars; Excel caps at '
                     f'{EXCEL_CELL_LIMIT:,} and would truncate them - write CSV instead')

    out = SUPP / fname
    with pd.ExcelWriter(out, engine='openpyxl') as xl:
        for sheet, df in sheets.items():
            df.to_excel(xl, sheet_name=sheet, index=False)
    print(f'  wrote {out} ({out.stat().st_size / 1e6:.1f} MB) sheets={list(sheets)}')


def main():
    SUPP.mkdir(exist_ok=True)
    print('supplement_table:', SUPP)

    print('\n== kinase_info ==')
    info = kdata.load('kinase_info')
    print('  ', info.shape, '| longest cell: {:,} chars in {}'.format(*max_cell_len(info)))
    write_csv(info, 'kinase_info.csv')

    print('\n== cddm ==')
    # index is the kinase; make it a normal column so each sheet is self-describing
    cddm = kdata.load('cddm').reset_index(names='kinase')
    cddm_LO = kdata.load('cddm_LO').reset_index(names='kinase')
    print('   freq', cddm.shape, '| log-odds', cddm_LO.shape,
          '| NaN cells in log-odds:', int(cddm_LO.isna().values.sum()))
    write_excel({'CDDM': cddm, 'CDDM_log_odds': cddm_LO}, 'cddm.xlsx')

    print('\n== ks_dataset ==')
    ks = kdata.ks_dataset(thr=None)
    print('  ', ks.shape, '| longest cell: {:,} chars in {}'.format(*max_cell_len(ks)))
    print('   full protein sequences included - this file is large and slow to write')
    write_csv(ks, 'ks_dataset.csv')


if __name__ == '__main__':
    main()
