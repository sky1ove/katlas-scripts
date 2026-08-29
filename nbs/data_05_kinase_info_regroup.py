"""data_05 · Promote `modi_group` to the canonical `group` in kinase_info.

Takes the annotated table from data_04 and renames

    group      -> group_old   (the original Coral grouping, kept for reference)
    modi_group -> group       (the curated Modi et al. grouping, now canonical)

then overwrites the `kinase_info` dataset. The version it replaces is kept in
`katlas_datasets/_archive/`.

This is destructive if applied twice - the second run would push the already-curated
`group` into `group_old` and lose the Coral grouping - so the script refuses to run
when the input already carries a `group_old` column.

Inputs   out/kinase_info_modi.csv (data_04)
Outputs  katlas_datasets/kinase_info.csv

Run:  python nbs/data_05_kinase_info_regroup.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from paths import OUT

import kdata


def main():
    src = OUT / 'kinase_info_modi.csv'
    if not src.exists():
        sys.exit(f'{src} not found - run data_04_modi_group.py first')

    info = pd.read_csv(src)
    print('dataset store:', kdata.DATASET)
    print('input:', src, info.shape)

    if 'group_old' in info.columns:
        sys.exit('input already has `group_old` - the regroup was applied before; '
                 'nothing to do (re-running would discard the Coral grouping)')
    if 'modi_group' not in info.columns:
        sys.exit('input has no `modi_group` column - run data_04_modi_group.py first')

    info = info.rename(columns={'group': 'group_old', 'modi_group': 'group'})
    changed = (info['group'] != info['group_old']).sum()
    print(f'{changed}/{len(info)} kinases change group')
    print(info[['kinase', 'group', 'group_old']].head().to_string())

    kdata.save('kinase_info', info)

    check = kdata.load('kinase_info')
    print('reload ->', check.shape,
          '| has group:', 'group' in check.columns,
          '| has group_old:', 'group_old' in check.columns)


if __name__ == '__main__':
    main()
