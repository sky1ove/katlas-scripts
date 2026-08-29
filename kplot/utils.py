"""General plotting helpers, save utilities, and palette/color tools."""


import itertools
from itertools import cycle
from pathlib import Path

import matplotlib as mpl
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statannotations.Annotator import Annotator

def set_sns(dpi: int = 300) -> None:
    "Set seaborn defaults for notebook display and saved figures."
    sns.set(rc={"figure.dpi": dpi, "savefig.dpi": dpi})
    sns.set_context('notebook')
    sns.set_style('ticks')


#: full print area of the target journal figure, in mm (width x height)
PRINT_W_MM, PRINT_H_MM = 180, 170


def paper_panel(frac: float = 1 / 2, ratio: float = 1.65,
                font: float = 7, legend_font: float = 6, line: float = 0.6) -> tuple[float, float]:
    """figsize (inches) + global font sizes for a panel taking `frac` of the fixed 180 mm figure width.

    The journal figure is a fixed 180 x 170 mm area. `frac` is the panel's share of the 180 mm width:
    pass a fraction such as 1/3, 1/2, 2/3 or 1 (full width). height = width / `ratio` (landscape when
    ratio > 1, default 1.65; pass e.g. 1.5 for a taller panel). Leave a little of the fraction for
    inter-panel gutters when placing panels side by side. Sets fonts to `font` pt (legend `legend_font`,
    a touch smaller) and axis/tick lines to `line` pt. Fonts are in points, so place the SVG at 100%
    (no rescaling). Save via save_svg.

        fig, ax = plt.subplots(figsize=paper_panel(1/2))             # half-width panel, ratio 1.65
        fig, ax = plt.subplots(figsize=paper_panel(2/3, ratio=2))   # two-thirds width, taller
    """
    w_mm = PRINT_W_MM * frac
    h_mm = w_mm / ratio
    plt.rcParams.update({'font.size': font, 'axes.titlesize': font, 'axes.labelsize': font,
                         'xtick.labelsize': font, 'ytick.labelsize': font, 'legend.fontsize': legend_font,
                         'axes.linewidth': line, 'xtick.major.width': line, 'ytick.major.width': line,
                         'xtick.minor.width': line, 'ytick.minor.width': line})
    return (w_mm / 25.4, h_mm / 25.4)

def save_svg(path: str | Path) -> None:
    "Save the current matplotlib figure as SVG with editable text."
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(path, format='svg', bbox_inches='tight', transparent=True)

def save_pdf(path: str | Path) -> None:
    "Save the current matplotlib figure as PDF with TrueType fonts."
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    plt.savefig(path, format='pdf', bbox_inches='tight', transparent=True)

def save_show(
    path: str | Path | None = None,  # output path when saving instead of showing
    show_only: bool = False,  # force plt.show even when no path is provided
) -> None:
    "Show the current figure or save it, then close open figures."
    if show_only:
        plt.show()
    elif path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0.05, transparent=True)
    else:
        plt.show()
    plt.close('all')

def get_color_dict(
    categories: list[str],  # labels that need colors
    palette: str = 'tab20',  # seaborn palette name
) -> dict[str, tuple[float, float, float]]:
    "Assign colors to labels while tolerating duplicate category names."
    colors = sns.color_palette(palette)
    color_cycle = cycle(colors)
    return {category: next(color_cycle) for category in categories}

def get_plt_color(
    palette: dict | list | str,  # dict lookup, explicit list, or palette name
    columns: list[str],  # plotted column names in output order
) -> list:
    "Return colors in plotting order for a dict, list, or named palette."
    if isinstance(palette, dict):
        return [palette.get(col, '#cccccc') for col in columns]
    if isinstance(palette, str):
        return sns.color_palette(palette, n_colors=len(columns))
    if isinstance(palette, list):
        return palette
    raise TypeError('palette must be a dict, list, or seaborn palette name')


def add_stats(ax,
              df,
              value,
              group,
              test='t-test_ind',
              loc='inside',
              text_format='star',
              min_n=3,
              **kwargs):
    """
    If `value` is str:
        compare between groups (x=group, y=value)
    If `value` is list/tuple:
        compare among values within each group (x=group, hue='variable')
    """

    # -----------------------------
    # Case 1: value is a single column -> compare groups
    # -----------------------------
    if isinstance(value, str):
        # get x axis labels from the plot
        order = [t.get_text() for t in ax.get_xticklabels()]

        # Build all between-group pairs, but skip too-small samples
        pairs = []
        for g1, g2 in itertools.combinations(order, 2):
            d1 = df.loc[df[group] == g1, value].dropna()
            d2 = df.loc[df[group] == g2, value].dropna()
            if len(d1) >= min_n and len(d2) >= min_n:
                pairs.append((g1, g2))

        if len(pairs) == 0:
            print('No valid group-vs-group comparisons (check sample sizes).')
            return ax

        annotator = Annotator(ax, pairs, data=df, x=group, y=value, order=order)
        annotator.configure(test=test, text_format=text_format, loc=loc,
                            verbose=False, **kwargs)
        annotator.apply_and_annotate()
        return ax

    # -----------------------------
    # Case 2: value is list/tuple -> compare values within each group
    # -----------------------------
    if not isinstance(value, (list, tuple)):
        raise TypeError('`value` must be a column name (str) or a list/tuple of column names.')

    df_melted = df.melt(id_vars=group, value_vars=list(value)).dropna(subset=['value'])

    pairs = []
    for g in df_melted[group].unique():
        sub = df_melted[df_melted[group] == g]
        vars_in_group = sub['variable'].unique()

        for v1, v2 in itertools.combinations(vars_in_group, 2):
            d1 = sub.loc[sub['variable'] == v1, 'value']
            d2 = sub.loc[sub['variable'] == v2, 'value']
            if len(d1) >= min_n and len(d2) >= min_n:
                pairs.append(((g, v1), (g, v2)))

    if len(pairs) == 0:
        print('No valid within-group comparisons (check sample sizes).')
        return ax

    annotator = Annotator(ax, pairs, data=df_melted, x=group, y='value', hue='variable')
    annotator.configure(test=test, text_format=text_format, loc=loc,
                        verbose=False, **kwargs)
    annotator.apply_and_annotate()
    return ax
