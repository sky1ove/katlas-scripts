"""Dimensionality-reduction, scatter, and correlation."""


from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from fastcore.meta import delegates
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap.umap_ import UMAP

def reduce_feature(
    df: pd.DataFrame,  # numeric feature matrix
    method: str = 'pca',  # one of pca, tsne, umap
    complexity: int = 20,  # perplexity for tsne or neighbors for umap
    n: int = 2,  # number of output dimensions
    load: str | Path | None = None,  # path to a previously fitted reducer
    save: str | Path | None = None,  # optional path for persisting the reducer
    seed: int = 123,  # random_state used by reducers that support it
    **kwargs,  # forwarded reducer kwargs
) -> pd.DataFrame:
    "Reduce a feature matrix to a lower-dimensional embedding dataframe."
    method = method.lower()
    if method not in {'pca', 'tsne', 'umap'}:
        raise ValueError('Please choose a method among PCA, TSNE, and UMAP')

    if load is not None:
        reducer = joblib.load(load)
        if hasattr(reducer, 'transform'):
            projection = reducer.transform(df)
        else:
            projection = reducer.fit_transform(df)
    else:
        if method == 'pca':
            reducer = PCA(n_components=n, random_state=seed, **kwargs)
        elif method == 'tsne':
            reducer = TSNE(n_components=n, random_state=seed, perplexity=complexity, **kwargs)
        else:
            reducer = UMAP(n_components=n, random_state=seed, n_neighbors=complexity, **kwargs)
        projection = reducer.fit_transform(df)

        if save is not None:
            save_path = Path(save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(reducer, save_path)

    embedding_df = pd.DataFrame(projection, index=df.index)
    embedding_df.columns = [f'{method.upper()}{i}' for i in range(1, embedding_df.shape[1] + 1)]
    return embedding_df

def plot_2d(
    embedding_df: pd.DataFrame,  # dataframe with at least two numeric columns
    hue: str | None = None,  # column name used for color when present in embedding_df
    palette: str = 'tab20',  # seaborn palette name
    legend: bool = False,  # whether to draw a legend
    name_list: list[str] | None = None,  # labels used to annotate points
    s: int = 20,  # marker size
    legend_title: str | None = None,  # optional legend title override
    legend_loc: str | None = None,  # None keeps seaborn's default; 'right'/'out'/'outside' places legend outside on the right
    **kwargs,  # forwarded scatterplot kwargs
):
    "Plot the first two columns of an embedding dataframe."
    if embedding_df.shape[1] < 2:
        raise ValueError('embedding_df must contain at least two columns to plot in 2D')
    if hue is not None and hue not in embedding_df.columns:
        raise ValueError(f'hue column {hue!r} not found in embedding_df')
    if name_list is not None and len(name_list) != len(embedding_df):
        raise ValueError('name_list must have the same length as embedding_df')

    x_col, y_col = embedding_df.columns[:2]
    fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (6, 4)))
    scatter_kwargs = dict(data=embedding_df, x=x_col, y=y_col, s=s, alpha=0.8, legend=legend, ax=ax, **kwargs)
    if hue is not None:
        scatter_kwargs['hue'] = hue
        scatter_kwargs['palette'] = palette
    sns.scatterplot(**scatter_kwargs)
    ax.set_xticks([])
    ax.set_yticks([])

    if legend and ax.legend_ is not None:
        # Keep the current title unless an override is given.
        cur_title = ax.legend_.get_title().get_text()
        title = legend_title if legend_title is not None else (cur_title or None)
        if legend_loc in ('right', 'out', 'outside'):
            # Place the legend outside, to the right, so it doesn't cover the points.
            sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1.02, 1),
                            borderaxespad=0, title=title, frameon=True)
        elif title is not None:
            ax.legend_.set_title(title)

    if name_list is not None:
        texts = [
            ax.text(embedding_df[x_col].iloc[i], embedding_df[y_col].iloc[i], str(name_list[i]), fontsize=8)
            for i in range(len(embedding_df))
        ]
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'))
    return ax


@delegates(sns.regplot)
def plot_rel(
    df: pd.DataFrame,  # dataframe that contains the x and y columns
    x: str,  # x-axis column name
    y: str,  # y-axis column name
    text_location: tuple[float, float] = (0.8, 0.1),  # annotation location in axes coordinates
    method: str | None = 'spearman',  # one of spearman, pearson, or None
    index_list: list[str] | None = None,  # row labels to annotate
    hue: str | None = None,  # optional categorical hue column
    reg_line: bool = True,  # whether to draw a regression line when hue is used
    figsize: tuple = (6, 4),  # figure size, used only when ax is None
    ax: plt.Axes | None = None,  # existing axis to draw on; a new figure is created when None
    **kwargs,  # forwarded seaborn kwargs
):
    "Plot a pairwise relationship with an optional correlation annotation."
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    x_vals = df[x]
    y_vals = df[y]

    if hue is not None:
        sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)
        if reg_line:
            sns.regplot(x=x_vals, y=y_vals, scatter=False, line_kws={'color': 'gray', 'alpha': 0.5}, ax=ax)
        if ax.legend_ is not None:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
    else:
        sns.regplot(x=x_vals, y=y_vals, line_kws={'color': 'gray'}, ax=ax, **kwargs)

    if method is not None:
        xv, yv = np.asarray(x_vals, float), np.asarray(y_vals, float)   # correlate finite pairs only
        _m = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[_m], yv[_m]
        if method.lower() == 'spearman':
            corr_val, pvalue = spearmanr(xv, yv)
            corr_label = f'Spearman ρ = {corr_val:.2f}\n p = {pvalue:.2e}'
        elif method.lower() == 'pearson':
            corr_val, pvalue = pearsonr(xv, yv)
            corr_label = f'Pearson r = {corr_val:.2f}\n p = {pvalue:.2e}'
        else:
            raise ValueError('method must be one of spearman, pearson, or None')
        ax.text(text_location[0], text_location[1], corr_label, transform=ax.transAxes, ha='center', va='center')

    texts = []
    if index_list is not None:
        for idx in index_list:
            if idx in df.index:
                texts.append(ax.text(x_vals.loc[idx], y_vals.loc[idx], str(idx), fontsize=9, ha='center', va='center'))
        if texts:
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5), ax=ax)
    return ax


def jointscatter(df, x, y, hue, palette,       # long-form data + the color-by column and its {level: color} map
                 figsize=(3.5, 2.7),           # whole-figure size (inches); e.g. kplot.utils.paper_panel(...)
                 xlabel=None, ylabel=None,      # axis labels (default: the column names)
                 lim=None,                      # shared x/y limits [lo, hi] (default: data range + 5% pad)
                 bins=25,                       # number of histogram bins over `lim`
                 s=6, alpha=0.85,               # scatter marker size / opacity
                 fit=True,                      # draw a least-squares best-fit line through the points
                 spearman=True,                 # annotate Spearman rho (top-left)
                 spearman_p=True,               # add the p-value on the line below rho
                 spearman_fontsize=7,           # font size of the Spearman annotation
                 legend=True,                   # draw a compact hue legend inside the scatter
                 legend_ncol=2, legend_loc='lower right',   # legend columns / location
                 labelpad=(2, 0),               # (x, y) label pad in points (small = tight tick-to-label gap)
                 tick_len=2, tick_pad=1.5,      # tick length / label pad (short ticks read better when small)
                 marg=(1, 4),                   # (marginal, main) grid ratio for the histogram panels
                 wspace=0.04, hspace=0.04):     # gaps between the main axes and the marginals
    """Joint scatter of `y` vs `x` colored by `hue`, with marginal histograms stacked by `hue`.

    A compact joint plot for method-comparison figures: a full box (all four spines), tight tick-to-label
    spacing, an optional least-squares fit line and Spearman annotation, and a small hue legend. Levels
    (and stacking/legend order) follow `palette`'s key order when it is a dict. Returns (fig, (ax, ax_top,
    ax_right)).
    """
    from matplotlib.lines import Line2D
    d = df.dropna(subset=[x, y, hue])
    levels = ([g for g in palette if g in set(d[hue])] if isinstance(palette, dict)
              else list(dict.fromkeys(d[hue])))
    col = (lambda g: palette[g]) if isinstance(palette, dict) else dict(zip(levels, palette)).__getitem__
    if lim is None:
        v = np.r_[d[x].to_numpy(float), d[y].to_numpy(float)]
        pad = 0.05 * (v.max() - v.min())
        lim = [v.min() - pad, v.max() + pad]
    edges = np.linspace(lim[0], lim[1], bins + 1)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[marg[1], marg[0]], height_ratios=[marg[0], marg[1]],
                          wspace=wspace, hspace=hspace)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)

    for g in levels:
        sub = d[d[hue] == g]
        ax.scatter(sub[x], sub[y], s=s, color=col(g), edgecolor='none', alpha=alpha)
    if fit:
        m, b = np.polyfit(d[x], d[y], 1)
        xf = np.array([d[x].min(), d[x].max()])
        ax.plot(xf, m * xf + b, color='0.4', lw=.9, zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(xlabel or x, labelpad=labelpad[0])
    ax.set_ylabel(ylabel or y, labelpad=labelpad[1])
    ax.tick_params(length=tick_len, pad=tick_pad)
    if spearman:
        res = spearmanr(d[x], d[y])
        txt = f'Spearman ρ = {res.correlation:.2f}'
        if spearman_p:                                        # p on the next line (e.g. 4.06e-60)
            txt += f'\np = {res.pvalue:.2e}'
        ax.text(.05, .95, txt, transform=ax.transAxes, fontsize=spearman_fontsize, va='top')
    if legend:
        ax.legend([Line2D([], [], marker='o', ls='', ms=3, color=col(g)) for g in levels], levels,
                  fontsize=5, frameon=False, ncol=legend_ncol, loc=legend_loc,
                  handletextpad=.2, columnspacing=.5, labelspacing=.25)

    colors = [col(g) for g in levels]
    axt.hist([d[d[hue] == g][x] for g in levels], bins=edges, stacked=True, color=colors, lw=0)
    axr.hist([d[d[hue] == g][y] for g in levels], bins=edges, stacked=True, orientation='horizontal',
             color=colors, lw=0)
    axt.axis('off'); axr.axis('off')
    return fig, (ax, axt, axr)
