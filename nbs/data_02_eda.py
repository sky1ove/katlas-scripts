"""data_02 · Exploratory analysis of the unified KS dataset.

1. Source overlap  - how many unique kinase-substrate pairs each pair of databases shares
                     (all sites, S/T sites only, Y sites only).
2. Phosphosite Venn - KS substrate sites vs the human phosphoproteome.
3. S/T/Y composition - stacked counts of KS pairs per source; Sugiyama is Y-heavy.
4. Acceptor + kinase-group pie charts.
5. Kinome-tree counts - per-kinase unique-pair counts, log10, for CORAL.

Inputs   out/combine_source.parquet, out/human_phosphoproteome.parquet  (data_01)
         kdata.ks_dataset(thr=EDA_THR)  - note 4/5 are filtered, 1-3 are not; see EDA_THR
Outputs  fig/EDA_*.pdf|svg, out/kinome_tree_cnt.csv

CORAL (http://phanstiel-lab.med.unc.edu/CORAL/) settings for the kinome tree:
  node color -> quantitative; paste out/kinome_tree_cnt.csv without the header;
  identifier coralID; min 0 max 3.5; manual 2-color low #E0E0E0 high #FA6958.
  Labels off: advanced setting -> label font size 0.

Run:  python nbs/data_02_eda.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib_venn import venn2
from paths import FIG, OUT

import kdata
from kplot.utils import get_color_dict, save_pdf, save_svg, set_sns

#: num_kin cut-off for the pie charts / kinome-tree counts. None = every pair, which is what
#: EDA should show; the bare `Data.ks_dataset()` in the original notebook silently applied 40.
EDA_THR = None

GROUP_ORDER = ['CMGC', 'AGC',       # blue
               'TK', 'TKL',         # orange
               'CAMK', 'STE',       # green
               'CK1', 'NEK',        # red
               'Atypical', 'Other',  # purple
               'RGC']


# ---------------------------------------------------------------- plot helpers


def plot_overlap(df_concat, source_col='source', id_col='kin_sub_site', figsize=(7, 5),
                 title='Overlap of Unique KS Pairs Between Sources'):
    "Lower-triangle heatmap of pairwise set intersections between sources (log colour scale)."
    source_ids = df_concat.groupby(source_col)[id_col].apply(set).to_dict()
    sources = list(source_ids)

    overlap = pd.DataFrame(index=sources, columns=sources, dtype=int)
    for s1 in sources:
        for s2 in sources:
            overlap.loc[s1, s2] = len(source_ids[s1] & source_ids[s2])
    overlap = overlap.astype(int)

    mask = np.triu(np.ones_like(overlap, dtype=bool), k=1)
    vmin, vmax = np.min(overlap[overlap > 0]), np.max(overlap)

    plt.figure(figsize=figsize)
    sns.heatmap(overlap, annot=True, mask=mask, fmt=',',
                norm=LogNorm(vmin=vmin, vmax=vmax), cmap='Blues',
                cbar=False, linewidths=1, linecolor='white')
    plt.title(title)


def plot_pie(value_counts, hue_order=None, labeldistance=0.8, fontsize=9,
             fontcolor='black', palette='tab20'):
    "Pie chart sorted by size; `hue_order` fixes the colour of each label."
    vc = value_counts.sort_values(ascending=False)
    labels = vc.index.tolist()
    if hue_order is not None:
        color_map = dict(zip(hue_order, sns.color_palette(palette, n_colors=len(hue_order))))
        colors = [color_map.get(lbl, 'grey') for lbl in labels]
    else:
        colors = sns.color_palette(palette, n_colors=len(labels))

    vc.plot.pie(autopct='%1.1f%%', labeldistance=labeldistance,
                textprops={'fontsize': fontsize, 'color': fontcolor}, colors=colors)
    plt.ylabel('')
    plt.title(f'n={value_counts.sum():,}')


def plot_stacked(df, figsize=(6, 4), with_legend=True):
    "Stacked count of KS pairs per source, coloured by phosphoacceptor."
    plt.figure(figsize=figsize)
    sns.histplot(data=df, x='source', hue='acceptor', multiple='stack', discrete=True,
                 shrink=0.8, alpha=1, palette=get_color_dict(['S', 'T', 'Y'], 'tab20'),
                 hue_order=['S', 'T', 'Y'])
    plt.xlabel('')
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    if with_legend:
        plt.ylabel('Number of KS Pairs')
        plt.title('Total Number of Unique KS Pairs per Source')
    else:
        # narrow companion panel for Sugiyama, which dwarfs the other sources
        plt.ylabel('')
        plt.legend([], frameon=False)
        plt.tick_params(axis='y', labelright=True, labelleft=False, right=True, left=False)


def add_sub_site(df):
    "Add `sub_site` (`<substrate_uniprot>_<site>`) and drop duplicates on it."
    df = df.copy()
    print('  original shape:', df.shape)
    df['sub_site'] = df['substrate_uniprot'] + '_' + df['site']
    df = df.drop_duplicates(subset='sub_site')
    print('  after dedup   :', df.shape)
    return df


# ---------------------------------------------------------------- analyses


def source_overlap(df):
    print('\n== Source overlap ==')
    plot_overlap(df)
    save_pdf(FIG / 'EDA_overlap.pdf')
    plt.close()

    plot_overlap(df[df.site.str[0] != 'Y'],
                 title='Overlap of KS Pairs in S/T sites Between Sources')
    save_pdf(FIG / 'EDA_overlap_ST.pdf')
    plt.close()

    plot_overlap(df[df.site.str[0] == 'Y'],
                 title='Overlap of KS Pairs in Y sites Between Sources')
    save_pdf(FIG / 'EDA_overlap_Y.pdf')
    plt.close()


def phosphosite_venn(df, human):
    print('\n== KS sites vs human phosphoproteome ==')
    human_site = add_sub_site(human)
    df_site = add_sub_site(df)

    set_human, set_ks = set(human_site.sub_site), set(df_site.sub_site)
    print('  human only:', len(set_human - set_ks),
          '| shared:', len(set_human & set_ks),
          '| KS only:', len(set_ks - set_human))

    plt.figure(figsize=(5, 5))
    venn = venn2([set_human, set_ks], set_labels=('', ''))
    for label in venn.subset_labels:
        if label:
            label.set_text(f'{int(label.get_text()):,}')
    plt.gca().set_aspect(0.8)
    plt.title('Overlap of Unique Substrate Sites Between Datasets')
    save_svg(FIG / 'EDA_venn_ks_human_overlap.svg')
    save_pdf(FIG / 'EDA_venn_ks_human_overlap.pdf')
    plt.close()


def sty_composition(df):
    print('\n== S/T/Y composition per source ==')
    df = df.copy()
    df['acceptor'] = df['site'].str[0]
    df_sugi = df[df.source == 'Sugiyama']
    df_rest = df[df.source != 'Sugiyama']
    print('  Sugiyama:', dict(df_sugi.acceptor.value_counts()))
    print('  rest    :', dict(df_rest.acceptor.value_counts()))

    plot_stacked(df_rest)
    save_svg(FIG / 'EDA_stacked_KS_per_source.svg')
    save_pdf(FIG / 'EDA_stacked_KS_per_source.pdf')
    plt.close()

    plot_stacked(df_sugi, figsize=(0.7, 4), with_legend=False)
    save_pdf(FIG / 'EDA_stacked_KS_per_source_part2.pdf')
    plt.close()


def dataset_pies():
    print('\n== Acceptor and kinase-group distribution ==')
    df = kdata.ks_dataset(thr=EDA_THR)
    print(f'  {len(df):,} pairs (num_kin cut-off: {EDA_THR})')
    df['acceptor'] = df.site.str[0]

    plot_pie(df['acceptor'].value_counts(), ['S', 'T', 'Y'])
    save_pdf(FIG / 'EDA_pie_STY.pdf')
    plt.close()

    plot_pie(df.kinase_group.value_counts(), hue_order=GROUP_ORDER)
    save_pdf(FIG / 'EDA_pie_group.pdf')
    plt.close()
    return df


def kinome_tree_counts(df):
    "Per-kinase log10 unique-pair count, in the CSV shape CORAL expects."
    print('\n== Kinome tree counts ==')
    cnt = (df.groupby('kinase_coral_ID')['kin_sub_site']
             .nunique().sort_values(ascending=False))
    out = OUT / 'kinome_tree_cnt.csv'
    cnt.apply(np.log10).round(2).to_csv(out)
    print('  wrote', out, '|', len(cnt), 'kinases')


def main():
    set_sns(100)
    df = pd.read_parquet(OUT / 'combine_source.parquet')
    human = pd.read_parquet(OUT / 'human_phosphoproteome.parquet')

    source_overlap(df)
    phosphosite_venn(df, human)
    sty_composition(df)
    ks = dataset_pies()
    kinome_tree_counts(ks)
    print('\nfigures written to', FIG)


if __name__ == '__main__':
    main()
