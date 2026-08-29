"""Categorical, count, distribution, and composition plot helpers."""


import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from fastcore.meta import delegates
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from .utils import add_stats, get_plt_color


@delegates(sns.barplot)
def plot_bar(
    df: pd.DataFrame,  # long-form dataframe
    value: str,  # numeric column name
    group: str,  # grouping column name
    title: str | None = None,  # optional plot title
    figsize: tuple[float, float] = (12, 5),  # figure size in inches
    fontsize: int = 14,  # axis label and tick size
    dots: bool = True,  # whether to overlay strip dots
    rotation: float = 90,  # x tick rotation angle
    ascending: bool = False,  # sort group means ascending when True
    ymin: float | None = None,  # optional lower y-axis bound
    **kwargs,  # forwarded barplot kwargs
):
    "Plot a bar chart from an unstacked dataframe." 
    fig, ax = plt.subplots(figsize=figsize)
    idx = df.groupby(group)[value].mean().sort_values(ascending=ascending).index
    sns.barplot(data=df, x=group, y=value, order=idx, hue=group, dodge=False, legend=False, ax=ax, **kwargs)

    if dots:
        marker = {'marker': 'o', 'color': 'white', 'edgecolor': 'black', 'linewidth': 1.5, 'jitter': True, 's': 5}
        sns.stripplot(data=df, x=group, y=value, order=idx, alpha=0.8, ax=ax, **marker)

    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_xlabel('')
    ax.set_ylabel(value, fontsize=fontsize)
    plt.xticks(rotation=rotation)
    if title is not None:
        plt.title(title, fontsize=fontsize)
    if ymin is not None:
        plt.ylim(bottom=ymin)
    ax.spines[['right', 'top']].set_visible(False)
    return ax

@delegates(sns.barplot)
def plot_group_bar(
    df: pd.DataFrame,  # wide-form dataframe
    value_cols: list[str],  # numeric columns to melt into grouped bars
    group: str,  # grouping column preserved during melt
    figsize: tuple[float, float] = (12, 5),  # figure size in inches
    order=None,  # optional x order passed to seaborn
    title: str | None = None,  # optional plot title
    fontsize: int = 14,  # axis label and tick size
    rotation: float = 90,  # x tick rotation angle
    **kwargs,  # forwarded barplot kwargs
):
    "Plot grouped bars after melting multiple value columns." 
    df_melted = df.melt(id_vars=group, value_vars=value_cols, var_name='Ranking', value_name='Value')
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=df_melted,
        x=group,
        y='Value',
        hue='Ranking',
        order=order,
        capsize=0.1,
        err_kws={'linewidth': 1.5, 'color': 'gray'},
        alpha=1.0,
        ax=ax,
        **kwargs,
    )
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_xlabel('')
    ax.set_ylabel('Value', fontsize=fontsize)
    plt.xticks(rotation=rotation)
    if title is not None:
        plt.title(title, fontsize=fontsize)
    ax.spines[['right', 'top']].set_visible(False)
    plt.legend(fontsize=fontsize, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    return ax


def plot_group_violin(
    df: pd.DataFrame,                 # long-form data (a row per observation)
    value: str,                       # numeric column
    group: str,                       # x-axis categorical column
    hue: str | None = None,           # split each group into side-by-side violins (e.g. method)
    order=None,                       # x category order
    hue_order=None,                   # hue level order
    palette=None,                     # {level: color} map or a seaborn palette
    ax=None,                          # draw on an existing axes (else make one)
    figsize: tuple[float, float] = (6, 3),  # figure size (in) when ax is None
    dots: bool = True,                # overlay per-observation dots, colored by hue
    dot_size: float = 2.5,            # dot marker size
    dot_alpha: float = 0.55,          # dot opacity
    violin_alpha: float = 0.35,       # violin fill opacity
    width: float = 0.85,             # total width of the violins at each x tick
    inner=None,                       # violin inner ('box', 'quartile', or None)
    linewidth: float = 0.5,           # violin outline width
    legend: bool = True,              # keep the hue legend
    ylabel: str | None = None,        # y-axis label override
    **kwargs,                         # forwarded to sns.violinplot
):
    "Grouped violins (one per hue level within each `group`) with per-observation dots overlaid."
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    vkw = dict(data=df, x=group, y=value, order=order, cut=0, width=width,
               linewidth=linewidth, inner=inner, ax=ax)
    if hue is not None:
        vkw.update(hue=hue, hue_order=hue_order, palette=palette, dodge=True)
    else:
        vkw.update(hue=group, palette=palette, dodge=False, legend=False)
    sns.violinplot(**vkw, **kwargs)
    for c in ax.collections:                          # soften the violin fill (before the dots)
        c.set_alpha(violin_alpha)
        c.set_edgecolor('none')
    if dots:
        skw = dict(data=df, x=group, y=value, order=order, size=dot_size,
                   alpha=dot_alpha, jitter=0.15, ax=ax, edgecolor='none', legend=False)
        if hue is not None:
            sns.stripplot(hue=hue, hue_order=hue_order, palette=palette, dodge=True, **skw)
        else:
            sns.stripplot(color='0.25', **skw)
    ax.set_xlabel('')
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.spines[['right', 'top']].set_visible(False)
    if not legend and ax.get_legend() is not None:
        ax.get_legend().remove()
    return ax


