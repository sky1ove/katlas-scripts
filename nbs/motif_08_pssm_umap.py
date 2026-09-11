"""motif_08 · UMAP of the CDDM, PSPA and MLP-attr PSSMs, coloured by kinase group.

A 2-D view of the same specificity space motif_06 clusters hierarchically - useful for
spotting whether kinase groups occupy distinct regions of substrate-preference space. One
UMAP each for the CDDM (substrate-derived) PSSMs, the scaled PSPA (peptide-array) PSSMs, and the
CDDM MLP-attribution PSSMs (out/mlp_attr_pssm_full.parquet, from motif_17; skipped if absent).
Pseudokinases are excluded.

Inputs   kdata: cddm, pspa_scale, kinase_info; out/mlp_attr_pssm_full.parquet (motif_17)
Outputs  fig/CDDM_umap.svg, fig/PSPA_umap.svg, fig/MLP_attr_umap.svg

Run:  python nbs/motif_08_pssm_umap.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import pandas as pd
from paths import FIG, OUT

import kdata
from katlas.utils import group_color
from kplot.scatter import plot_2d, reduce_feature
from kplot.utils import save_svg

MLP_ATTR = OUT / 'mlp_attr_pssm_full.parquet'


def umap_fig(pssm, info, title, out):
    "UMAP one PSSM table, colour by kinase group, save to `out`."
    embed = reduce_feature(pssm, method='umap', complexity=6, min_dist=0.5)
    embed['group'] = embed.index.map(info.set_index('kinase')['group'])
    print(f'  {title}: {pssm.shape} | groups:', dict(embed.group.value_counts(dropna=False)))

    plot_2d(embed, s=40, figsize=(4, 4), hue='group', legend=True, palette=group_color,
            legend_title='Group', legend_loc='right')
    plt.title(title)
    save_svg(out)
    plt.close('all')
    print('  wrote', out)


def main():
    info = kdata.load('kinase_info')
    info = info[info.pseudo == '0']
    print('non-pseudo kinases:', len(info))

    umap_fig(kdata.load('cddm'), info, 'UMAP of CDDM PSSM', FIG / 'CDDM_umap.svg')
    pspa_u = kdata.load('pspa_scale').dropna(axis=1)
    pspa_u = pspa_u[~pspa_u.index.astype(str).str.endswith('_TYR')]   # drop the 15 dual-spec _TYR duplicates
    umap_fig(pspa_u, info, 'UMAP of PSPA PSSM', FIG / 'PSPA_umap.svg')

    if MLP_ATTR.exists():                                   # signed attribution matrix from motif_17
        umap_fig(pd.read_parquet(MLP_ATTR).fillna(0.0), info, 'UMAP of MLP-attr PSSM',
                 FIG / 'MLP_attr_umap.svg')
    else:
        print(f'skip MLP-attr UMAP: {MLP_ATTR} not found (run motif_17)')


if __name__ == '__main__':
    main()
