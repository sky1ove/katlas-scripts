"""data_03 · Site promiscuity - how many kinases target the same substrate site?

1. Distribution  - bin the ~30k substrate sites by distinct-kinase count (1, 2~10,
                   11~100, 101~300). About half the sites have a single known kinase;
                   ~0.2% are hit by 100+ kinases.
2. Motifs        - sequence logo per promiscuity bin.
3. Source split  - highly promiscuous sites come almost entirely from Sugiyama.
4. Acceptor split- Y sites are over-represented in the 11~100 bin.
5. Gene groups   - the 101~300 bin is enriched for glycolysis, actin cytoskeleton,
                   heat-shock and ribosomal proteins.

Finally overwrites the `ks_unique` dataset: one row per substrate site with `num_kin`,
`bin`, `source_combine` and a one-hot kinase-binding matrix. The version it replaces is
kept in `katlas_datasets/_archive/`.

Inputs   kdata.ks_dataset(thr=None), raw/genes_grouped.csv
Outputs  katlas_datasets/CDDM/unique_ks_sites.parquet, fig/promi_*.pdf|svg

Run:  python nbs/data_03_site_promiscuity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from paths import FIG, RAW

import kdata
from katlas.plot import plot_logo, plot_pssm_heatmaps
from katlas.pssm import get_IC, get_prob
from katlas.utils import sty_color
from kplot.utils import paper_panel, save_pdf, save_svg, set_sns

BINS = [0, 1, 10, 100, 300]
LABELS = ['1', '2~10', '11~100', '101~300']

# Panel widths as a share of the 180 mm figure width: the histogram takes a little under
# half, the two composition panels split the remainder. Each panel is w = 180 mm * FRAC
# wide and w / ratio high, so the three do not come out the same height at these ratios.
HIST_FRAC, COMP_FRAC = 0.47, 0.26
HIST_RATIO, COMP_RATIO = 1.75, 0.75
HIST_FONT = 7          # the histogram carries its key inside the axes, at the body size
LOGO_FRAC, LOGO_RATIO = 1.0, 2.5   # the stacked bin logos take the full figure width

# Promiscuity thresholds used to show one kinase's motif sharpening as shared sites drop
# out. 40 is the cut the motif pipeline actually applies (motif_01 reads ks_dataset(thr=40)).
FILTER_KINASE = 'ABL1'
FILTER_THRS = [None, 80, 40, 20, 10, 5]
FILTER_WIN = 5                     # +-5 flank, the window the PSSM store uses


# ---------------------------------------------------------------- plot helpers


def plot_pie(value_counts, hue_order=None, labeldistance=0.8, fontsize=12,
             fontcolor='black', palette='tab20', figsize=(4, 3)):
    if hue_order is not None:
        value_counts = value_counts.reindex(hue_order)
    colors = sns.color_palette(palette, n_colors=len(value_counts))
    value_counts.plot.pie(autopct='%1.1f%%', labeldistance=labeldistance,
                          textprops={'fontsize': fontsize, 'color': fontcolor},
                          colors=colors, figsize=figsize)
    plt.ylabel('')
    plt.title(f'n={value_counts.sum():,}')


def plot_cnt(cnt, xlabel=None, ylabel='Count', figsize=(6, 3)):
    "Bar chart with the count printed on top of each bar."
    fig, ax = plt.subplots(figsize=figsize)
    cnt.plot.bar(ax=ax)
    for idx, value in enumerate(cnt):
        ax.text(idx, value + 0.5, f'{value:,}', ha='center', va='bottom', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.xticks(rotation=0)
    plt.tight_layout()


def get_pct(df, bin_col, hue_col):
    "Row-normalised (percent) cross-tab of `bin_col` x `hue_col`."
    count_df = df.groupby([bin_col, hue_col], observed=False).size().unstack(fill_value=0)
    return count_df.div(count_df.sum(axis=1), axis=0) * 100


def get_plt_color(palette, columns):
    "Resolve a dict / list / seaborn palette name into a colour list matching `columns`."
    if isinstance(palette, dict):
        return [palette.get(col, '#cccccc') for col in columns]
    if isinstance(palette, str):
        return sns.color_palette(palette, n_colors=len(columns))
    return palette


def plot_composition(df, bin_col, hue_col, palette='tab20', title=None, rotate=45,
                     xlabel=None, ylabel='Percentage', figsize=None):
    """Stacked percentage bar of `hue_col` composition within each `bin_col` level.

    The categories are named by a keyless legend laid out in one row just under the panel
    title: a vertical box beside the axes would eat a third of the width at this size, and
    the panel title already says which variable the categories belong to.
    """
    pct_df = get_pct(df, bin_col, hue_col)
    # rc_context: paper_panel sets the paper font sizes globally, and the pie / bar / logo
    # figures later in this script are not paper panels and should keep their own sizes
    with plt.rc_context():
        fig, ax = plt.subplots(figsize=figsize or paper_panel(COMP_FRAC, ratio=COMP_RATIO))
        pct_df.plot(kind='bar', ax=ax, stacked=True, width=0.75,
                    color=get_plt_color(palette, pct_df.columns))
        ax.set_ylabel(ylabel, labelpad=2)
        ax.set_xlabel(xlabel, labelpad=2)
        ax.tick_params(axis='both', length=2, pad=1.5)
        # the bin labels do not fit side by side at this panel width, so they are angled
        # and anchored at their right edge to sit under their own tick
        plt.setp(ax.get_xticklabels(), rotation=rotate, ha='right' if rotate else 'center',
                 rotation_mode='anchor' if rotate else None)
        ax.set_title(title, pad=14)
        ax.legend(ncol=len(pct_df.columns), loc='lower center', bbox_to_anchor=(0.5, 1.0),
                  frameon=False, columnspacing=1, handlelength=1, handletextpad=0.4,
                  borderpad=0, handleheight=0.9)
        plt.tight_layout()


# ---------------------------------------------------------------- analyses


def build_sites(df):
    "One row per substrate site with its distinct-kinase count, bin, gene names and site_seq."
    num_kin = df.groupby('sub_site')['kinase_uniprot'].nunique()
    binned = pd.cut(num_kin, bins=BINS, labels=LABELS, right=True, include_lowest=True)

    sites = pd.concat([num_kin, binned], axis=1)
    sites.columns = ['num_kin', 'bin']
    sites = sites.reset_index()

    uniq = df.drop_duplicates('sub_site').set_index('sub_site')
    sites['sub_genes'] = sites.sub_site.map(uniq.substrate_genes)
    sites['site_seq'] = sites.sub_site.map(uniq.site_seq)
    sites['gene'] = sites.sub_genes.str.split(' ').str[0]
    sites['acceptor'] = sites.sub_site.str.split('_').str[1].str[0]

    print('  sites:', sites.shape)
    print('  per bin:', dict(sites.bin.value_counts().sort_index()))
    return sites


def plot_num_kin_hist(sites, palette='Pastel1', figsize=None):
    """Histogram of distinct kinases per substrate site, one bar per integer count.

    `num_kin` is a count, so the bins are unit width (no fractional edges smearing
    neighbouring counts together). The distribution spans 1 to a few hundred with half
    the mass on the single bar at 1, so the y-axis is log; bars are coloured by the
    promiscuity bin used everywhere else in this script.
    """
    n = sites.num_kin
    colors = get_plt_color(palette, LABELS)
    bin_of = pd.cut(n, bins=BINS, labels=LABELS, right=True, include_lowest=True)

    with plt.rc_context():   # paper font sizes stay local to this panel, see plot_composition
        fig, ax = plt.subplots(figsize=figsize or paper_panel(
            HIST_FRAC, ratio=HIST_RATIO, font=HIST_FONT, legend_font=HIST_FONT))
        edges = range(1, int(n.max()) + 2)
        for label, color in zip(LABELS, colors):
            sub = n[bin_of == label]
            share = len(sub) / len(n) * 100
            ax.hist(sub, bins=edges, color=color, edgecolor='none',
                    label=f'{label} ({share:.1f}%)')

        ax.set_yscale('log')
        ax.set_xlabel('Kinases per substrate site', labelpad=2)
        ax.set_ylabel('Substrate sites', labelpad=2)
        ax.set_title(f'n = {len(n):,} sites; median = {n.median():.0f}, '
                     f'max = {n.max():.0f}')
        ax.tick_params(length=2, pad=1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # paper_panel does not set legend.title_fontsize, which would otherwise fall back
        # to the global default and print the key's title far larger than its entries
        ax.legend(title='Promiscuity bin', frameon=False,
                  title_fontsize=plt.rcParams['legend.fontsize'])
        plt.tight_layout()
        save_svg(FIG / 'promi_hist.svg')
        plt.close()


def plot_distribution(sites):
    print('\n== Kinases per site distribution ==')
    plot_num_kin_hist(sites)

    cnt = sites.bin.value_counts()
    plot_pie(cnt, fontsize=9, labeldistance=1, palette='Pastel1')
    save_pdf(FIG / 'promi_pie.pdf')
    plt.close()

    plot_cnt(cnt.sort_index(), xlabel='# Kinases per Substrate Site',
             ylabel='# Substrate Sites', figsize=(5, 2.4))
    save_pdf(FIG / 'promi_bar.pdf')
    plt.close()


def stack_logos(mats, figsize, sharey=False, xlabel=None, xticks=None, ylabel='IC (bits)'):
    """Stack one logo per entry of `mats` ({title: frequency PSSM}) in a single column.

    `sharey` puts every row on one information-content scale. That is what makes heights
    comparable, so use it only when the rows have comparable n: the IC estimator is biased
    upward by about (A - 1) / (2 N ln2) bits per position, so a small row would otherwise
    tower over the rest on estimation noise alone.
    """
    fig, axes = plt.subplots(len(mats), 1, figsize=figsize, sharex=True, sharey=sharey)
    axes = np.atleast_1d(axes)
    for i, (ax, (name, m)) in enumerate(zip(axes, mats.items())):
        plot_logo(m, title='', ax=ax)
        ax.set_title(name, loc='left', pad=2)
        # plot_logo labels every position and drops to a 10 pt condensed font on a narrow
        # axis; both are wrong for a paper panel, so the ticks are set again here
        if xticks is not None:
            ax.set_xticks(xticks)
        plt.setp(ax.get_xticklabels(), fontsize=plt.rcParams['xtick.labelsize'],
                 fontfamily=plt.rcParams['font.family'])
        ax.tick_params(length=2, pad=1.5)
        ax.set_ylabel(ylabel if i == len(mats) // 2 else '', labelpad=2)
    if sharey:
        # every column of a frequency PSSM sums to 1, so a logo column's total height is
        # exactly that position's IC. logomaker leaves the axis at its own default, which
        # is far above the data here, so set it from the tallest column across the rows.
        ymax = max(get_IC(m).max() for m in mats.values())
        axes[0].set_ylim(0, ymax * 1.08)
    axes[-1].set_xlabel(xlabel, labelpad=2)
    plt.tight_layout()
    return fig, axes


def plot_bin_logos(sites, figsize=None, sharey=False):
    """Sequence logo of each promiscuity bin, stacked in one column on a shared x axis.

    One row per bin over the same +-20 window, so the flanking preferences can be read
    down the column. Each row keeps its own information-content scale: the bins differ in
    size by more than two orders of magnitude (14,600 vs 72 sites), and the IC estimator
    is biased upward by roughly (A - 1) / (2 N ln2) bits per position, which is 0.001 bits
    at n = 14,600 but 0.22 bits at n = 72. Sharing one y axis would therefore flatten the
    three large bins against a small bin whose height is substantially estimation noise.
    Read each row's own motif, not the heights across rows.
    """
    print('\n== Motif per promiscuity bin ==')
    mats = {}
    for b in LABELS:
        sites_b = sites[sites.bin == b]
        if not len(sites_b):
            continue
        kin_str = 'kinase' if b == '1' else 'kinases'
        mats[f'{b} {kin_str} (n = {len(sites_b):,})'] = get_prob(sites_b, 'site_seq')

    with plt.rc_context():
        stack_logos(mats, figsize or paper_panel(LOGO_FRAC, ratio=LOGO_RATIO), sharey=sharey,
                    xlabel='Position relative to the phospho-acceptor',
                    xticks=range(-20, 21, 5))
        save_svg(FIG / 'promi_logos.svg')
        plt.close()


def kinase_filter_motifs(df, kinase=FILTER_KINASE, thrs=FILTER_THRS, win=FILTER_WIN):
    """One kinase's motif rebuilt at a series of promiscuity cutoffs, as logos and heatmaps.

    Dropping sites that many kinases share leaves the ones that kinase is actually selective
    for, so its determinant positions stand out. Note the mean information content barely
    moves: what sharpens is a few positions, while the rest get noisier as n falls (the IC
    estimator's upward bias grows as 1 / n). Read the determinant cells, not the overall
    height, and treat the low-count rows as the noisier ones.
    """
    print(f'\n== {kinase} motif vs promiscuity cutoff ==')
    sub = df[df.kinase_genes.str.split(' ').str[0] == kinase].copy()
    # motif_01 builds the CDDM after a case-insensitive dedup, so mirror it here
    sub = sub.assign(seq_upper=sub.site_seq.str.upper()).drop_duplicates('seq_upper')

    mats, counts = {}, {}
    for t in thrs:
        keep = sub if t is None else sub[sub.num_kin <= t]
        label = 'no filter' if t is None else f'$\\leq${t}'
        pssm = get_prob(keep, 'site_seq')
        mats[label] = pssm[[c for c in pssm.columns if abs(c) <= win]]
        counts[label] = len(keep)
    print('  sites per cutoff:', counts)

    titled = {f'{k} (n = {counts[k]:,})': v for k, v in mats.items()}
    with plt.rc_context():
        # Flank only: position 0 is the acceptor (Y for a tyrosine kinase), the same in every
        # row and tall enough to take most of a shared y axis, which would squash the flanking
        # positions that the filter is actually changing. It stays in the heatmaps.
        # sharey: n only falls 4x across the series (1,616 -> 379), so the IC bias gap
        # between the rows (0.010 -> 0.042 bits) is small next to the change being shown
        flanks = {k: v[[c for c in v.columns if c != 0]] for k, v in titled.items()}
        stack_logos(flanks, paper_panel(1 / 2, ratio=0.8), sharey=True,
                    xlabel='Position relative to the phospho-acceptor (0 omitted)',
                    xticks=[c for c in range(-win, win + 1) if c != 0])
        save_svg(FIG / f'promi_{kinase}_logos.svg')
        plt.close()

    with plt.rc_context():
        # one colour scale across the row, otherwise each panel is normalised to its own
        # maximum and every cutoff looks equally sharp
        flank_max = max(m[[c for c in m.columns if c != 0]].to_numpy().max() for m in mats.values())
        plot_pssm_heatmaps(mats, figsize=paper_panel(0.7, ratio=2.6), vmax=flank_max,
                           cmap='Reds', wspace=0.09,
                           title=f'{kinase} motif by kinases per site')
        save_svg(FIG / f'promi_{kinase}_pssms.svg')
        plt.close()


def promiscuous_gene_groups(sites):
    "Which protein groups carry the most promiscuous (101~300 kinases) sites?"
    print('\n== Gene groups of the most promiscuous sites ==')
    groups = pd.read_csv(RAW / 'genes_grouped.csv')
    # drop the "(small)" / "(large)" qualifier on ribosomal proteins
    groups.Group = groups.Group.str.split('(').str[0]
    group_map = groups.set_index('Gene')['Group'].to_dict()

    top = sites[sites.bin == '101~300'].copy()
    top['gene_group'] = top.gene.map(group_map)
    print(top.gene_group.value_counts().to_string())


def add_source_combine(sites, df):
    "Label each site Sugiyama / Non-Sugiyama / Both, from the `|`-joined source tags."

    def convert_source(x):
        if x == 'Sugiyama':
            return x
        return 'Both' if ('Sugiyama' in x and '|' in x) else 'Non-Sugiyama'

    def combine_source(sources):
        sources = set(sources)
        if sources == {'Sugiyama'}:
            return 'Sugiyama'
        if sources == {'Non-Sugiyama'}:
            return 'Non-Sugiyama'
        return 'Both'

    src = df.assign(source2=df.source.apply(convert_source)).groupby('sub_site')['source2'].unique()
    sites = sites.copy()
    sites['source_combine'] = sites.sub_site.map(src).apply(combine_source)
    return sites


def source_composition(sites):
    "Split each bin by whether its sites come from Sugiyama, elsewhere, or both."
    print('\n== Composition by source ==')
    print(get_pct(sites, 'bin', 'source_combine').round(1).to_string())

    plot_composition(sites, 'bin', 'source_combine', palette='Set2', title='Source composition',
                     xlabel='Kinases per substrate site')
    save_svg(FIG / 'promi_bar_source_percentage.svg')
    plt.close()


def acceptor_composition(sites):
    print('\n== Composition by acceptor ==')
    plot_composition(sites, 'bin', 'acceptor', palette=sty_color, title='Acceptor composition',
                     xlabel='Kinases per substrate site')
    save_svg(FIG / 'promi_bar_site_percentage.svg')
    plt.close()

    print(get_pct(sites, 'bin', 'acceptor').round(1).to_string())


def save_sites(sites, df):
    "Attach a one-hot kinase-binding matrix and overwrite the `ks_unique` dataset."
    print('\n== Save unique sites ==')
    df = df.copy()
    df['kinase_uniprot_gene'] = (df['kinase_uniprot'] + '_'
                                 + df['kinase_genes'].str.split(' ').str[0])
    pivot = pd.crosstab(df['sub_site'], df['kinase_uniprot_gene']).reset_index()
    sites = sites.merge(pivot)
    print('  with one-hot kinases:', sites.shape)

    kdata.save('ks_unique', sites, index=False)


def main():
    set_sns(100)

    df = kdata.ks_dataset(thr=None)
    print('ks_dataset:', df.shape)

    sites = add_source_combine(build_sites(df), df)
    plot_distribution(sites)
    plot_bin_logos(sites)
    kinase_filter_motifs(df)
    promiscuous_gene_groups(sites)
    source_composition(sites)
    acceptor_composition(sites)
    save_sites(sites, df)


if __name__ == '__main__':
    main()
