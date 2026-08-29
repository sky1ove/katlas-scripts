"""plot functions of probability pssm"""


from typing import Callable

from fastcore.meta import delegates
import numpy as np, pandas as pd
from .pssm import get_IC, recover_pssm
from .utils import pSTY2sty, sty2pSTY, sty2pSTY_df

from matplotlib import pyplot as plt
from matplotlib import font_manager
import logomaker, math

#: first available narrow/condensed font for dense tick labels (tighter glyphs + tighter minus);
#: None -> fall back to the default font. Order = most-portable first, then the user's preference.
_NARROW_FONT = next(
    (f for f in ('DejaVu Sans Condensed', 'Roboto Condensed', 'Arial Narrow',
                 'PT Sans Narrow', 'DIN Condensed', 'Avenir Next Condensed')
     if f in {ff.name for ff in font_manager.fontManager.ttflist}),
    None,
)
import seaborn as sns

from matplotlib.colors import TwoSlopeNorm

# for plot two heatmaps
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter

@delegates(sns.heatmap)
def plot_heatmap(heatmap_df, 
                 ax=None, 
                 position_label=True, 
                 figsize=(5, 6), 
                 include_zero=True,
                 scale_pos_neg=False, 
                 colorbar_title='prob',
                 **kwargs
                 ):
    "Plot a heatmap of pssm."
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    heatmap_df = sty2pSTY_df(heatmap_df)
    mask = np.zeros_like(heatmap_df, dtype=bool)
    zero_position = len(heatmap_df.columns) // 2
    second_position = math.ceil(len(heatmap_df.columns) / 2)
    # If they overlap, move the second line one step to the right (if possible)
    if second_position == zero_position: second_position = second_position + 1

    if not include_zero:
        mask[:, zero_position] = True  # Mask position 0 if include_zero is False

    cmap = plt.get_cmap('coolwarm').copy()
    cmap.set_bad('white')  # NaN cells (e.g. low-count masked) render white, distinct from the value-0 colour

    if scale_pos_neg:
        vmin,vmax = heatmap_df.min().min(),heatmap_df.max().max()
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        sns.heatmap(
            heatmap_df,
            cmap=cmap,
            norm=norm,
            linewidth=0.3,
            ax=ax,
            mask=mask,
            **kwargs
        )
    else:
        sns.heatmap(
            heatmap_df,
            cmap=cmap,
            center=0,  # Center for diverging colormap
            linewidth=0.3,
            ax=ax,
            mask=mask,
            **kwargs
        )


    # Access and format the color bar
    colorbar = ax.collections[0].colorbar
    colorbar.ax.set_title(colorbar_title, loc='center')

    # Add vertical lines
    ax.axvline(zero_position, color='black', linewidth=0.5)
    ax.axvline(second_position, color='black', linewidth=0.5)

    # Format the heatmap border
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1.5)

    # Hide axis labels
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.xaxis.set_ticks_position('top')
    ax.set_yticks(np.arange(len(heatmap_df)) + 0.5)          # label every residue: seaborn auto-thins the
    ax.set_yticklabels(heatmap_df.index, rotation=0)         # y ticks to every other one on a small figure
    if not position_label:
        ax.set_xticklabels([])

    return ax


def plot_logo_raw(pssm_df,ax=None,title='Motif',ytitle='Bits',figsize=(10,2)):
    "Plot logo motif using Logomaker."
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # convert s,t,y to pS,pT,pY for visualization
    pssm_df=sty2pSTY_df(pssm_df)

    logo = logomaker.Logo(pssm_df.T, color_scheme='kinase_protein', flip_below=False, ax=ax)
    logo.ax.set_ylabel(ytitle)
    # label every position; if the axis is too narrow to fit default-size numbers, drop to a small
    # condensed font — decided from the ACTUAL axis width, so large figures keep the default font
    ax_w_in = ax.get_position().width * ax.figure.get_figwidth()
    tick_kw = ({'fontsize': 10, 'fontfamily': _NARROW_FONT}
               if _NARROW_FONT and ax_w_in * 72 / pssm_df.shape[1] < 16 else {})
    logo.style_xticks(anchor=0, spacing=1, fmt='%d', **tick_kw)
    ax.set_title(title)

def prepare_logo_df(df):
    "Prepare a PSSM for logo plotting by moving center s/t/y labels and replacing missing values with 0."
    df = df.copy()

    # find which of s/t/y actually exist
    lowercase = [aa for aa in ['s', 't', 'y'] if aa in df.index]

    # check if any of them have non-zero value at position 0 (a flank-only matrix has no
    # position 0 at all, and then there is no centre label to move)
    if lowercase and 0 in df.columns and (df.loc[lowercase, 0] != 0).any():
        for aa in lowercase:
            upper_aa = aa.upper()
            if upper_aa not in df.index:
                df.loc[upper_aa] = 0
            df.loc[upper_aa, 0] = df.loc[aa, 0]
            df.loc[aa, 0] = 0

    return df.fillna(0)

def get_pos_min_max(pssm_df):
    """
    Get min and max value of sum of positive and negative values across each position.
    """
    pssm_df = pssm_df.copy()
    pssm_neighbor = pssm_df.drop(columns=0)
    
    max_sum_pos = pssm_neighbor[pssm_neighbor>0].sum().max()
    max_sum_neg = pssm_neighbor[pssm_neighbor<0].sum().min()
    return max_sum_neg,max_sum_pos

def scale_zero_position(pssm_df):
    """
    Scale position 0 so that:
    - Positive values match the max positive column sum of other positions
    - Negative values match the min (most negative) column sum of other positions
    """
    max_sum_neg,max_sum_pos = get_pos_min_max(pssm_df)

    zero_col = pssm_df[0]
    zero_col_pos = zero_col[zero_col>0]
    zero_col_neg = zero_col[zero_col<0]
    
    scaled_col = zero_col.copy()
    if not zero_col_pos.empty and zero_col_pos.sum() != 0:
        scaled_col.loc[zero_col_pos.index] = max_sum_pos * (zero_col_pos / zero_col_pos.sum())
    if not zero_col_neg.empty and zero_col_neg.sum() != 0:
        scaled_col.loc[zero_col_neg.index] = max_sum_neg * (zero_col_neg / zero_col_neg.sum())

    pssm_df[0] = scaled_col
    return pssm_df
    

def scale_pos_neg_values(pssm_df):
    """
    Globally scale all positive values by max positive column sum,
    and negative values by min negative column sum (preserving sign).
    """
    pssm_df = pssm_df.copy()
    max_sum_neg, max_sum_pos = get_pos_min_max(pssm_df)

    pos_part = pssm_df.clip(lower=0)
    neg_part = pssm_df.clip(upper=0)

    if max_sum_pos != 0: pos_part = pos_part / max_sum_pos
    if max_sum_neg != 0: neg_part = neg_part / abs(max_sum_neg)  # make sure sign is correct

    return pos_part + neg_part

def convert_logo_df(pssm_df,scale_zero=True,scale_pos_neg=False):
    "Prepare a PSSM for logo plotting and optionally scale the zero position or signed values."
    pssm_df = prepare_logo_df(pssm_df)
    # a flank-only matrix has no position 0 to rescale (and get_pos_min_max would fail
    # dropping it), so the acceptor step is simply skipped
    if scale_zero and 0 in pssm_df.columns: pssm_df = scale_zero_position(pssm_df)
    if scale_pos_neg: pssm_df = scale_pos_neg_values(pssm_df)
    return pssm_df

def get_logo_IC(pssm_df, **kwargs):
    """
    For plotting purpose, calculate the scaled information content (bits) from a frequency matrix,
    using log2(3) for the middle position and log2(len(pssm_df)) for others. `keep_center=True`
    keeps a fully-conserved centre residue visible.
    """
    IC_position = get_IC(pssm_df, **kwargs)

    return pssm_df.mul(IC_position, axis=1) # total_IC = pssm_df.sum().sum().round(2)

def plot_logo(pssm_df,title='Motif', scale_zero=True,ax=None,figsize=(10,1),keep_center=False):
    "Plot logo of information content given a frequency PSSM. `keep_center` shows a fixed centre residue."
    pssm_df = get_logo_IC(pssm_df, keep_center=keep_center)
    pssm_df= convert_logo_df(pssm_df,scale_zero=scale_zero)
    plot_logo_raw(pssm_df,ax=ax,title=title,ytitle='IC (bits)',figsize=figsize)


def plot_logos(pssms_df, 
               count_dict=None, # used to display n in motif title
               prefix='Motif',
               figsize=(14,1)
               ):
    """
    Plot all logos from a dataframe of flattened PSSMs as subplots in a single figure.
    """
    n = len(pssms_df)
    hspace=0.7
    # 14 is width, 1 is height for each logo
    width,height=figsize
    fig, axes = plt.subplots(nrows=n, figsize=(width, n * (height+hspace)),gridspec_kw={'hspace': hspace+0.1})

    if n == 1:
        axes = [axes]  # ensure axes is iterable

    for ax, idx in zip(axes, pssms_df.index):
        pssm = recover_pssm(pssms_df.loc[idx])
        if count_dict is not None:
            plot_logo(pssm, title=f"{prefix or ''} {idx} (n={count_dict[idx]:,})",ax=ax)
        else:
            plot_logo(pssm, title=f"{prefix or ''} {idx}",ax=ax)

def _plot_logo_heatmap_layout(
    heatmap_df,  # Matrix to render in the lower panel
    logo_plotter: Callable,  # Function that draws the logo on the provided axis
    figsize: tuple = (17, 10),  # Figure size
    include_zero: bool = False,  # Whether to show the center position in the heatmap
    heatmap_kwargs: dict | None = None,  # Extra keyword arguments passed to plot_heatmap
    square: bool = False,  # auto-size the figure width from the matrix shape so heatmap cells render square
    hspace: float = 0.11,  # vertical gap between the logo and the heatmap (raise it to unclamp the position numbers)
    height_ratios=(1, 5),  # logo:heatmap height split (raise the first to make the logo taller / heatmap flatter)
):
    "Plot a logo above a heatmap using the shared katlas layout."
    heatmap_kwargs = {} if heatmap_kwargs is None else dict(heatmap_kwargs)

    if square:
        # The heatmap axis occupies fixed fractions of the figure (~0.62 width, ~0.607 height, set by the
        # gridspec + colorbar below), so width = height*(ncol/nrow)*(yfrac/xfrac) makes each cell square
        # while the heatmap stays aligned with the logo above. (Only exact for the default height_ratios.)
        nrow, ncol = heatmap_df.shape
        figsize = (figsize[1] * (ncol / nrow) * (0.607 / 0.62), figsize[1])

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=list(height_ratios), width_ratios=[4, 1], hspace=hspace, wspace=0)

    ax_logo = fig.add_subplot(gs[0, 0])
    logo_plotter(ax_logo)

    ax_heatmap = fig.add_subplot(gs[1, :])

    plot_heatmap(heatmap_df, ax=ax_heatmap, position_label=False, include_zero=include_zero, **heatmap_kwargs)
    return fig, ax_logo, ax_heatmap


def style_logo_heatmap(fig,                 # a logo+heatmap figure from `_plot_logo_heatmap_layout` (or the
                                            # `plot_logo_heatmap*` plotters): axes are [logo, heatmap, colorbar?]
                       positions=None,      # x labels to place UNDER the heatmap (e.g. [-5..+4]); None keeps them
                       ylabel=None,         # replace the logo y-axis label, pulled in (e.g. 'log2')
                       title=None,          # replace the logo title; None just tightens the existing one
                       border_lw=0.6,       # uniform heatmap border width (drawn as 4 equal spines)
                       tick_len=1.5,        # tick length for the logo y-axis / heatmap / colorbar
                       pos_fontsize=None,   # position-number size; default matches the heatmap residue labels
                       ):
    """Compact 'paper' restyling of a logo+heatmap figure: move the position numbers UNDER the heatmap so
    the logo sits flush on top, match the position-number size to the residue labels, shorten the ticks,
    draw a uniform thin border, pull the y-label in, and tighten the title. Returns the figure. Use it after
    a `plot_logo_heatmap*` call (e.g. `plot_logo_heatmap_pspa(row, square=False, ...)` then
    `style_logo_heatmap(plt.gcf(), positions=[-5,...,4], ylabel='log2', title='ERK1 - PSPA')`)."""
    axl, axh = fig.axes[0], fig.axes[1]
    axc = fig.axes[2] if len(fig.axes) > 2 else None
    if ylabel is not None: axl.set_ylabel(ylabel, labelpad=1)
    axl.set_xticks([])                                                    # position numbers move under heatmap
    axl.tick_params(axis='y', length=tick_len, pad=1)
    axl.set_title(title if title is not None else axl.get_title(), pad=2, fontsize=axl.title.get_fontsize())
    if positions is not None:
        fs = pos_fontsize or axh.get_yticklabels()[0].get_fontsize()     # match the residue-label size
        axh.xaxis.set_ticks_position('bottom'); axh.xaxis.set_label_position('bottom')
        axh.set_xticks(np.arange(len(positions)) + 0.5)
        axh.set_xticklabels(list(positions), fontsize=fs)
    axh.tick_params(axis='x', length=tick_len, pad=1, top=False, labeltop=False, bottom=True, labelbottom=True)
    axh.tick_params(axis='y', length=tick_len, pad=1)
    if axc is not None: axc.tick_params(length=tick_len, pad=1)
    axh.patch.set_linewidth(0)                                            # uniform border: 4 equal spines,
    for s in axh.spines.values():                                        # not the uneven thick patch edge
        s.set_visible(True); s.set_linewidth(border_lw); s.set_color('black')
    return fig

def plot_logo_heatmap(pssm_df, # column is position, index is aa
                       title='Motif',
                       figsize=(17,10),
                       include_zero=False,
                       square=False, # auto-size the width from the matrix shape so heatmap cells render square
                       hspace=0.11, # vertical gap between the logo and the heatmap
                       keep_center=False, # keep a fixed centre residue visible in the logo (e.g. the SD pTyr)
                      ):
    """Plot logo and heatmap vertically. `square=True` sizes the width from the matrix shape so cells are square."""
    _plot_logo_heatmap_layout(
        heatmap_df=pssm_df,
        logo_plotter=lambda ax: plot_logo(pssm_df, ax=ax, title=title, keep_center=keep_center),
        figsize=figsize,
        include_zero=include_zero,
        square=square,
        hspace=hspace,
    )


def plot_pssm_heatmaps(pssms,                       # dict {title: pssm}, a single pssm, or a list of pssms
                       aa_order=None,               # amino acids on the y-axis, top->bottom (default standard order)
                       positions=None,              # positions on the x-axis (default: those shared by all pssms)
                       drop_zero=False,             # skip the position-0 acceptor column, drawing a vertical
                                                    # divider line where it was (between -1 and +1)
                       divider_lw=0.6,              # line width of the -1/+1 divider when drop_zero is True
                       signed=None,                 # subset drawn on a diverging map (auto from negatives if None)
                       scale_acceptor=True,         # rescale position-0 to the flank scale: True = the
                                                    # distribution (non-signed) maps, False = none, or a list
                                                    # of names to rescale (even signed ones, e.g. CDDM/PSPA)
                       fill_ps_from_pt=None,        # names whose flank `s` row copies `t` (PSPA duplicate)
                       relabel_sty=True,            # show s/t/y as pS/pT/pY on the y-axis
                       figsize=None,                # whole-figure size (in); overrides `panel`
                       panel=(1.0, 2.0),            # (width, height) inches per heatmap when figsize is None
                       wspace=0.04,                 # gap between heatmaps
                       title=None,                  # overall figure title (e.g. the kinase / species)
                       title_y=None,                # title y (figure fraction); None = auto, a fixed
                                                    # absolute gap so stretching/flattening leaves it put
                       title_pad=0.33,              # inches reserved at the top for `title` (aspect-independent)
                       tight=True,                  # run tight_layout to pack the heatmaps
                       pad=0.2,                     # tight_layout padding (fraction of the font size)
                       cmap='viridis',              # colormap for the non-signed (distribution) heatmaps
                       cmap_signed='RdBu_r',        # diverging colormap for signed (e.g. MLP-attr) heatmaps
                       title_fontsize=6.5,          # per-heatmap title font size (overall `title` is +1)
                       tick_fontsize=5,             # x / y tick label font size
                       spine_lw=0.5,                # heatmap border line width
                       vmax=None,                   # fixed colour-scale limit shared by every heatmap
                                                    # (signed ones use +-vmax); None = per-heatmap, which
                                                    # normalises each matrix away and makes the panels
                                                    # look equally strong however they actually differ
                       cbar=False):                 # draw a colorbar beside each heatmap
    """One or more PSSMs as a row of heatmaps (amino acid x position); returns (fig, axes).

    Each pssm is a (residue x position) matrix (e.g. from `recover_pssm`) or a flat Series (auto-recovered).
    Every matrix is reindexed to `aa_order` rows and `positions` columns, so anything a matrix lacks shows
    as a blank cell. Only the leftmost heatmap draws the amino-acid labels.

    Scaling suits substrate PSSMs, where position 0 (the acceptor) dominates: `signed` matrices (default:
    any with negative values, e.g. MLP-attribution) use a symmetric diverging map; the rest are scaled to
    the flank, and `scale_acceptor` rescales the position-0 column on its own so its strongest cell reaches
    the top of the scale and the others sit in proportion (keeps 0s vs 0t legible). `fill_ps_from_pt` copies
    the flank `t` row into `s` for the named matrices (PSPA measures pT; its flanking pS is a duplicate).
    By default each heatmap is scaled to its own maximum, which is what you want when comparing motif
    shapes but not when comparing strengths: pass `vmax` to put every panel on one colour scale (e.g. a
    series of the same kinase built from progressively fewer sites).

    Tune `figsize`/`panel`/`wspace` for size and inter-heatmap gaps.
    """
    if isinstance(pssms, (pd.DataFrame, pd.Series)):
        pssms = {'': pssms}
    elif isinstance(pssms, (list, tuple)):
        pssms = {str(i): p for i, p in enumerate(pssms)}
    mats = {k: (recover_pssm(p) if isinstance(p, pd.Series) else p.copy()) for k, p in pssms.items()}

    if aa_order is None:
        aa_order = list('PGACSTVILMFYWHKRQNDE') + ['s', 't', 'y']
    if positions is None:
        positions = sorted(set.intersection(*[set(m.columns) for m in mats.values()]))
    if drop_zero:
        positions = [p for p in positions if p != 0]
    if signed is None:
        signed = {k for k, m in mats.items() if np.nanmin(m.to_numpy(dtype=float)) < -1e-9}
    elif signed is True:
        signed = set(mats)
    elif not signed:
        signed = set()
    else:
        signed = set(signed)
    fill_ps = set(fill_ps_from_pt or [])
    if scale_acceptor is True:                                 # default: the distribution (non-signed) matrices
        scale_set = {k for k in mats if k not in signed}
    elif not scale_acceptor:
        scale_set = set()
    else:                                                      # an explicit collection of names (even if signed)
        scale_set = set(scale_acceptor)

    for k, m in mats.items():
        if k in fill_ps and 's' in m.index and 't' in m.index:
            fl = [c for c in m.columns if c != 0]
            m.loc['s', fl] = m.loc['t', fl].values
        mats[k] = m.reindex(index=aa_order, columns=positions)

    n = len(mats)
    if figsize is None:
        figsize = (panel[0] * n, panel[1])
    fig, axes = plt.subplots(1, n, figsize=figsize, gridspec_kw={'wspace': wspace}, squeeze=False)
    axes = axes[0]
    ylab = [{'s': 'pS', 't': 'pT', 'y': 'pY'}.get(a, a) if relabel_sty else a for a in aa_order]
    flank = [c for c in positions if c != 0]
    for ax, (name, m) in zip(axes, mats.items()):
        disp = m.copy()
        do_scale = name in scale_set and 0 in positions        # rescale position-0 to the flank scale?
        # base the colour scale on the flank when rescaling the acceptor, so a large position-0 column
        # does not inflate vmax; the acceptor column is then scaled so its top cell (the larger of S/T)
        # reaches that same vmax, keeping the S/T ratio.
        base = m[flank] if (do_scale and flank) else m
        if name in signed:
            vhi = vmax if vmax is not None else np.nanmax(np.abs(base.to_numpy()))
            kw = dict(cmap=cmap_signed, vmin=-vhi, vmax=vhi)
        else:
            vhi = vmax if vmax is not None else np.nanmax(base.to_numpy())
            kw = dict(cmap=cmap, vmin=0, vmax=vhi)
        if do_scale:
            pm = np.nanmax(disp[0].to_numpy())
            if pm and pm > 0:
                disp[0] = disp[0] * (vhi / pm)
        sns.heatmap(disp, ax=ax, cbar=cbar, rasterized=True, **kw)
        ax.set_title(name, fontsize=title_fontsize, pad=2)
        ax.set_xlabel(''); ax.set_ylabel('')
        ax.set_xticks(np.arange(len(positions)) + 0.5, positions, fontsize=tick_fontsize)
        ax.tick_params(bottom=True, length=1.5, pad=1)
        if ax is axes[0]:
            ax.set_yticks(np.arange(len(aa_order)) + 0.5, ylab, fontsize=tick_fontsize, rotation=0)
            ax.tick_params(left=True, length=1.5, pad=1)
        else:
            ax.tick_params(left=False, labelleft=False)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(spine_lw)
        if drop_zero:                                          # mark where position 0 was, between -1 and +1
            ndiv = sum(1 for p in positions if p < 0)
            if 0 < ndiv < len(positions):
                ax.axvline(ndiv, color='0.15', lw=divider_lw, zorder=5)
    if tight:
        fig.tight_layout(pad=pad)
    # reserve a fixed absolute band (title_pad inches) at the top for the title, so its distance from
    # the heatmaps stays put when the figure is stretched or flattened. subplots_adjust(top=) is used
    # rather than tight_layout's rect, which heatmap axes ignore.
    if title is not None:
        head = min(0.30, title_pad / figsize[1])
        fig.subplots_adjust(top=1 - head)
        y = title_y if title_y is not None else 1 - head * 0.40
        fig.suptitle(title, fontsize=title_fontsize + 1, fontweight='bold', y=y)
    return fig, axes
