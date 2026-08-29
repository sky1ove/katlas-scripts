"""data_04 · Attach the curated Modi et al. kinase grouping to `kinase_info`.

Modi et al. (https://www.nature.com/articles/s41598-019-56499-4, supplementary table 5)
re-annotate the kinome groups and correct several assignments in the Coral grouping.
This script maps their `1_Group` onto `kinase_info` by UniProt accession and writes an
annotated copy with an extra `modi_group` column.

UniProt accessions that appear more than once in the Modi table are dropped (they were
matched by hand afterwards); kinases absent from the Modi table get NaN.

`data_05_kinase_info_regroup.py` is what promotes `modi_group` to the canonical `group`.

HISTORICAL: the regroup was already applied to the shipped `kinase_info`, which therefore
carries `group_old` and no `modi_group`. So data_04's output already has `group_old` and
data_05 will refuse to run - this pair only does anything on a pre-regroup `kinase_info`.

Inputs   raw/modi_group_41598_2019_56499_MOESM5_ESM.xlsx, kdata.load('kinase_info')
Outputs  out/kinase_info_modi.csv (never overwrites the packaged kinase_info)

Run:  python nbs/data_04_modi_group.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from paths import OUT, RAW

import kdata


def load_modi_group():
    "Modi table S5, one row per unambiguous UniProt accession."
    group = pd.read_excel(RAW / 'modi_group_41598_2019_56499_MOESM5_ESM.xlsx')
    dup = group[group['5_Uni_acc'].duplicated(keep=False)]
    print('dropping', len(dup), 'rows with duplicated UniProt accession:',
          sorted(dup['5_Uni_acc'].unique()))

    group = group[~group['5_Uni_acc'].duplicated(keep=False)]
    group = group[['1_Group', '2_Gene', '5_Uni_acc']]
    group.columns = ['modi_group', 'modi_gene', 'uniprot']
    return group


def main():
    OUT.mkdir(exist_ok=True)

    group = load_modi_group()
    info = kdata.load('kinase_info')
    print('kinase_info:', info.shape, '| modi table:', group.shape)

    group_id = group.set_index('uniprot')['modi_group']
    info['modi_group'] = info.uniprot.map(group_id)

    matched = info.modi_group.notna().sum()
    print(f'matched {matched}/{len(info)} kinases')
    print('unmatched kinases:', sorted(info[info.modi_group.isna()].kinase))
    print('in Modi but not on the kinome tree:',
          sorted(group[~group.uniprot.isin(info.uniprot)].modi_gene))

    # keep modi_group next to the existing group column for easy comparison
    cols = [c for c in info.columns if c != 'modi_group']
    cols.insert(cols.index('group') + 1, 'modi_group')
    info = info[cols]

    out = OUT / 'kinase_info_modi.csv'
    info.to_csv(out, index=False)
    print('wrote', out)


if __name__ == '__main__':
    main()
