"""motif_16 · How do CDDM, PSPA and surface display agree — across every representation?

Six per-kinase specificity matrices from three methods, in their frequency and their signed
(enrichment / log-odds) forms:

  CDDM  freq (`cddm`) · log-odds (`cddm_LO`) · MLP-attr (`out/mlp_attr_pssm_full`)
  PSPA  (`pspa`)   — one row: norm and enrich are identical under a rank metric (enrich is a
                     per-position monotonic transform of norm), so only one is shown.
  SD    freq (`sd_freq_p2`) · enrich (`sd_ptyrvar_p2`)

Agreement is scored per kinase over the ±5 flank (central acceptor at 0 excluded) with two
complementary metrics, then averaged over kinases:

  spearman   mean per-position Spearman of the score profile — whole-distribution agreement,
             invariant to the scale/centering differences between methods
  top5       overlap of the 5 strongest (position, residue) hotspots across the whole flank,
             |top5(a) ∩ top5(b)| / 5 — agreement on the residues that define the motif

Residues compared, per pair (the maximal set both methods measure): the 20 standard amino acids,
plus, when both sides carry phospho-priming, the primed residues pS (`s`), pT (`t`) and pY (`y`) kept
as separate features. CDDM/MLP encode pS and pT separately; PSPA cannot distinguish them, so its `s`
is a duplicate of `t` (both the measured pT), and the metrics run on the flank (position 0 excluded),
where that duplication is faithful, rather than collapsing the pair. This keeps a real pS-vs-pS /
pT-vs-pT comparison for pairs that both resolve pS/pT (e.g. CDDM vs MLP-attr). Surface display has no
primed residues, so any pair involving it has no `s`/`t`/`y` and falls back to the 20 standard residues.

Fig 5 (the surface-display figure) is a deliberate exception. Because it lines up PSPA, CDDM and
surface display side by side, and surface display has no priming, every method there is scored on the
20 standard AAs (via `scored_on(AA20)`) so all rows share one fair footing. The by-group vs-PSPA set
(`compare_vs_pspa_bygroup.csv`, feeding Fig 3) keeps the priming residues.

Two views:
  1. method × method  — every pair, on the tyrosine kinases all four methods share (n = 12).
  2. vs PSPA          — each method against PSPA as the reference, both on those 12 kinases and on
                        each method's own full overlap with PSPA (CDDM/MLP reach ~290-326, SD 12).

Inputs   kdata: cddm, cddm_LO, pspa; pssm/{sd_freq_p2,sd_ptyrvar_p2}.parquet;
         out/mlp_attr_pssm_full.parquet (motif_17)
Outputs  out/compare_methods.csv, out/compare_methods_matrix.csv, out/compare_vs_pspa.csv,
         fig/compare_methods_{spearman,top5}.svg, fig/compare_vs_pspa.svg, fig/compare_vs_pspa_violin_{spearman,ap}.svg,
         fig/compare_vs_pspa_perkinase_{spearman,top5,ap}.svg,
         fig/method_example_{ABL1,SRC}_{freq,signed}.svg, fig/compare_cddm_vs_pspa_{spearman,top5,ap}_bygroup.svg

Run:  python nbs/motif_16_compare_methods.py
"""
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from paths import FIG, OUT, PSSM
from scipy.stats import spearmanr

import kdata
from katlas.plot import plot_pssm_heatmaps
from katlas.pssm import recover_pssm
from katlas.utils import group_color
from kplot.bar import plot_group_bar, plot_group_violin
from kplot.utils import paper_panel, save_svg, set_sns

AA20 = list('ACDEFGHIKLMNPQRSTVWY')
FEATURES = AA20 + ['s', 't', 'y']               # default alphabet: +pS/pT/pY kept separate; a pair uses the subset
                                                # both have (PSPA's `s` duplicates `t`; the flank-only metric is
                                                # faithful to that). Fig 5 (with surface display) swaps to 20 AA only
                                                # via scored_on(AA20) so every method there is on one fair footing.
FLANK = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]     # ±5, central acceptor (0) excluded
MIN_RES = 5                                       # skip a position with < this many scored residues
TOPK = 5


@contextmanager
def scored_on(feats):
    "Temporarily score on residue alphabet `feats`; every metric reads the module-global FEATURES."
    global FEATURES
    saved, FEATURES = FEATURES, list(feats)
    try:
        yield
    finally:
        FEATURES = saved

MLP_ATTR = OUT / 'mlp_attr_pssm_full.parquet'

#: display label -> flat-PSSM source, grouped by method (CDDM's three, then PSPA, then SD's two).
METHODS = [
    ('CDDM freq',     'cddm'),
    ('CDDM log-odds', 'cddm_LO'),
    ('CDDM MLP-attr', str(MLP_ATTR)),
    ('PSPA',          'pspa'),
    ('SD freq',       'sd_freq_p2'),
    ('SD enrich',     'sd_ptyrvar_p2'),
]
KDATA_SETS = {'cddm', 'cddm_LO', 'pspa'}


# ---------------------------------------------------------------- matrices + metrics


def load_matrix(name):
    "Flat per-kinase PSSM table by source: kdata dataset, the pssm/ store, or an out/ parquet path."
    if name in KDATA_SETS:
        return kdata.load(name)
    if name.endswith('.parquet'):
        return pd.read_parquet(name)
    return pd.read_parquet(PSSM / f'{name}.parquet')


def recover(row):
    "Recover a flat PSSM row into an aa x position matrix; primed residues s/t/y are kept as separate rows."
    return recover_pssm(row.dropna())               # no priming collapse: FEATURES keeps s/t/y separate,
                                                    # and the flank-only metric is faithful to PSPA's s==t


def _cols(a, b):
    return [c for c in FLANK if c in a.columns and c in b.columns]


def perpos_spearman(a, b):
    "Mean Spearman over flank positions, on the features both matrices carry (per-position finite mask)."
    rs = []
    for c in _cols(a, b):
        x = a[c].reindex(FEATURES).to_numpy(float)
        y = b[c].reindex(FEATURES).to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= MIN_RES:
            rs.append(spearmanr(x[m], y[m]).correlation)
    return float(np.nanmean(rs)) if rs else np.nan


def top5_overall(a, b, k=TOPK):
    "Overlap of the k strongest (position, feature) cells over the whole flank, on cells both matrices have."
    cols = _cols(a, b)
    A = a.reindex(index=FEATURES, columns=cols)
    B = b.reindex(index=FEATURES, columns=cols)
    both = A.notna() & B.notna()                     # only cells measured by both count
    fa, fb = A.where(both).stack(), B.where(both).stack()
    if len(fa) < k or len(fb) < k:
        return np.nan
    return len(set(fa.nlargest(k).index) & set(fb.nlargest(k).index)) / k


def score(a, b):
    return {'spearman': perpos_spearman(a, b), 'top5': top5_overall(a, b)}


def _shared_cells(m, ref):
    "Method scores mv and reference gains rv over the (position, feature) cells both matrices measure."
    cols = _cols(m, ref)
    M = m.reindex(index=FEATURES, columns=cols)
    R = ref.reindex(index=FEATURES, columns=cols)
    both = M.notna() & R.notna()
    return M.where(both).stack().dropna(), R.where(both).stack().dropna()


def average_precision(m, ref, k=TOPK):
    "AP with PSPA's top-k cells as the relevant set, ranking all shared flank cells by m's score."
    mv, rv = _shared_cells(m, ref)
    if len(mv) <= k or rv.max() <= 0:
        return np.nan
    relevant = set(rv.nlargest(k).index)                                # PSPA's top-k hotspots
    hits, precs = 0, []
    for i, cell in enumerate(mv.sort_values(ascending=False).index, 1):
        if cell in relevant:
            hits += 1
            precs.append(hits / i)                                       # precision at each hit
    return float(np.mean(precs)) if precs else np.nan


# ---------------------------------------------------------------- comparisons


def shared_kinases():
    "Tyrosine kinases every method covers."
    return sorted(set.intersection(*(set(load_matrix(src).index) for _, src in METHODS)))


def compare(shared):
    "Per-kinase spearman + top5 for every method pair over the shared kinases."
    labels = [lbl for lbl, _ in METHODS]
    pssm = {lbl: {k: recover(load_matrix(src).loc[k]) for k in shared} for lbl, src in METHODS}
    rows = []
    for i, A in enumerate(labels):
        for B in labels[i:]:
            for k in shared:
                rows.append({'a': A, 'b': B, 'kinase': k, **score(pssm[A][k], pssm[B][k])})
    return pd.DataFrame(rows), labels


def matrices(res, labels):
    "Symmetric method×method mean-over-kinases matrices for each metric."
    out = {}
    for metric in ('spearman', 'top5'):
        M = pd.DataFrame(np.nan, index=labels, columns=labels)
        for (a, b), g in res.groupby(['a', 'b']):
            M.loc[a, b] = M.loc[b, a] = g[metric].mean()
        out[metric] = M
    return out


def vs_pspa(restrict=None):
    "Per-kinase spearman + top5 of each non-PSPA method vs PSPA; restrict = kinase subset (None = full overlap)."
    pspa = load_matrix('pspa')
    pm = {k: recover(pspa.loc[k]) for k in pspa.index}
    rows = []
    for lbl, src in METHODS:
        if lbl == 'PSPA':
            continue
        m = load_matrix(src)
        ks = set(pspa.index) & set(m.index)
        if restrict is not None:
            ks &= set(restrict)
        for k in sorted(ks):
            a = recover(m.loc[k])
            rows.append({'method': lbl, 'kinase': k, **score(a, pm[k]),
                         'ap': average_precision(a, pm[k])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

#: axis/title label per metric. Every one of these is already stated "vs PSPA", so the label must not
#: repeat PSPA; AP@5 is the name used throughout the paper for recovery of PSPA's 5 strongest cells.
METRIC_TITLE = {'spearman': 'Spearman', 'top5': 'Top-5 overlap', 'ap': 'AP@5'}
VS_METRICS = ['spearman', 'top5', 'ap']                # PSPA-reference views (ap is directional)
UNIT_METRICS = {'spearman', 'top5'}                    # bounded to [0, 1]; ap auto-scales
NON_PSPA = [lbl for lbl, _ in METHODS if lbl != 'PSPA']


def _annot(v):
    "2-dp cell label with the leading zero stripped ('.45', '-.02') to save width; exact 1 as '1'."
    if not np.isfinite(v):
        return ''
    if abs(v - 1) < 1e-9:
        return '1'
    t = f'{v:.2f}'
    return t.replace('0.', '.', 1) if t.startswith(('0.', '-0.')) else t


#: rows/cols of the method x method panels: (label in METHODS, label drawn). CDDM log-odds is still
#: computed and persisted (motif_20 and the HTML summary use it), just left off this panel, and the
#: CDDM prefixes are dropped since every non-SD row is CDDM-derived.
MATRIX_ROWS = [('CDDM freq', 'CDDM'), ('CDDM MLP-attr', 'MLP-attr'), ('PSPA', 'PSPA'),
               ('SD freq', 'SD freq'), ('SD enrich', 'SD enrich')]

#: rows of the per-kinase panels: the same display list minus PSPA (which is the reference there).
VS_ROWS = [(k, lbl) for k, lbl in MATRIX_ROWS if k != 'PSPA']

#: method x method panels: metric -> (title, output stem)
MATRIX_PANELS = {'spearman': ('Mean per-position Spearman', 'compare_methods_spearman'),
                 'top5': ('Top-5 hotspot overlap', 'compare_methods_top5')}


def plot_matrix(mats, n, frac=0.30, ratio=1.6, annot_fontsize=6, cbar=False, square=False,
                title_fontsize=8):
    """One paper-sized method x method heatmap per metric (Spearman, top-5), saved separately so each
    can be placed on its own. The 1.0 diagonal clips to a darker-than-vmax shade; cell labels drop the
    leading zero so they fit the small cells. `n` (shared kinases) belongs in the legend, not the panel.
    No colorbar by default: every cell is annotated, so the bar only re-encodes what is already readable
    and costs panel width. Note in the legend that the color scale is per panel.
    `square` forces square cells, which pins the aspect and makes `ratio` a no-op (it then only pads
    whitespace); leave it False for `ratio` to actually flatten the panel."""
    keys = [k for k, _ in MATRIX_ROWS]
    labels = [lbl for _, lbl in MATRIX_ROWS]
    diag = np.eye(len(keys), dtype=bool)
    reds = plt.get_cmap('Reds').with_extremes(over='#330000')
    for metric, (title, stem) in MATRIX_PANELS.items():
        m = mats[metric].loc[keys, keys]
        vmax = float(np.nanmax(m.values[~diag]))
        fig, ax = plt.subplots(figsize=paper_panel(frac, ratio=ratio))
        sns.heatmap(m, annot=np.vectorize(_annot)(m.values), fmt='', annot_kws={'fontsize': annot_fontsize},
                    cmap=reds, vmin=0, vmax=vmax, square=square, linewidths=0.4, cbar=cbar,
                    cbar_kws={'shrink': 0.6, 'extend': 'max'} if cbar else None, ax=ax)
        ax.set_title(title, fontsize=title_fontsize, pad=3)
        ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=7)
        ax.set_yticklabels(labels, rotation=0, fontsize=7)
        ax.tick_params(length=2, pad=1.5)
        if cbar:
            ax.collections[0].colorbar.ax.tick_params(labelsize=6, length=2, pad=1.5)
        save_svg(FIG / f'{stem}.svg')
        plt.close(fig)


def plot_vs_pspa_bars(ref12, refmax):
    "Bar grid: agreement with PSPA, rows = {12 shared kinases, each method's own overlap}, cols = metrics."
    fig, axes = plt.subplots(2, len(VS_METRICS), figsize=(4.6 * len(VS_METRICS), 8), constrained_layout=True)
    for row, (ref, tag) in enumerate([(ref12, '12 shared tyrosine kinases'),
                                      (refmax, "each method's own PSPA overlap")]):
        g = ref.groupby('method')
        mean, err = g[VS_METRICS].mean().reindex(NON_PSPA), g[VS_METRICS].std().reindex(NON_PSPA)
        nn = g['kinase'].nunique().reindex(NON_PSPA)
        for col, metric in enumerate(VS_METRICS):
            ax = axes[row, col]
            ax.bar(range(len(NON_PSPA)), mean[metric], yerr=err[metric], color='#b2182b',
                   capsize=3, error_kw={'lw': 0.8})
            ax.set_xticks(range(len(NON_PSPA)))
            ax.set_xticklabels(NON_PSPA, rotation=25, ha='right')
            if metric in UNIT_METRICS:
                ax.set_ylim(0, 1)
            else:
                lo = min(0.0, float((mean[metric] - err[metric]).min()))     # AP auto-scales (non-unit metric)
                ax.set_ylim(lo, max(float((mean[metric] + err[metric]).max()) * 1.15, 0.1))
                ax.axhline(0, color='k', lw=0.5)
            ax.set_title(f'{METRIC_TITLE[metric]} vs PSPA — {tag}')
            ax.spines[['top', 'right']].set_visible(False)
            for xi, mth in enumerate(NON_PSPA):
                ax.text(xi, -0.13, f'n={int(nn[mth])}', ha='center', va='top', fontsize=7, color='0.4',
                        transform=ax.get_xaxis_transform())
    save_svg(FIG / 'compare_vs_pspa.svg')
    plt.close('all')


def plot_vs_pspa_violin(ref12, frac=1 / 3, ratio=1.5, metrics=('spearman', 'ap')):
    """Per-metric summary of agreement with PSPA over the shared tyrosine kinases: a violin of the
    per-kinase distribution with one dot per kinase, replacing the mean+s.d. bar (which hid that
    n = 12 and that the spread is wide). Each method is shaded by its own mean through the same Reds
    map the heatmaps use, so darker still reads as better agreement. `n` goes in the legend. Both
    panels auto-scale to their own data: Spearman runs slightly negative here, so a fixed 0-1 axis
    would silently clip points."""
    keys = [k for k, _ in VS_ROWS]
    labels = dict(VS_ROWS)
    d = ref12[ref12.method.isin(keys)].copy()
    d['method'] = pd.Categorical(d.method.map(labels), [labels[k] for k in keys], ordered=True)
    reds = plt.get_cmap('Reds')
    for metric in metrics:
        mean = d.groupby('method', observed=True)[metric].mean()
        lo, hi = float(mean.min()), float(mean.max())
        pal = {m: reds(0.30 + 0.55 * ((v - lo) / (hi - lo) if hi > lo else 0.5)) for m, v in mean.items()}
        fig, ax = plt.subplots(figsize=paper_panel(frac, ratio=ratio))
        plot_group_violin(d, metric, 'method', hue=None, palette=pal, order=list(mean.index), ax=ax,
                          dot_size=2.5, violin_alpha=.45, legend=False)
        lo, hi = float(d[metric].min()), float(d[metric].max())   # every panel auto-scales, same rule:
        pad = 0.08 * (hi - lo)                                    # a fixed 0-1 range would clip the
        ax.set_ylim(lo - pad, hi + pad)                            # negative Spearman points
        if lo < 0:
            ax.axhline(0, color='0.6', lw=0.5, zorder=0)
        ax.set_ylabel(f'{METRIC_TITLE[metric]} vs PSPA', labelpad=2)
        ax.set_xlabel('')
        ax.set_xticklabels([labels[k] for k in keys], rotation=30, ha='right')
        ax.tick_params(length=2, pad=1.5)
        ax.spines[['top', 'right']].set_visible(False)
        save_svg(FIG / f'compare_vs_pspa_violin_{metric}.svg')
        plt.close(fig)


#: the example panels: display label -> loader. Each matrix is drawn in its OWN units. The first three
#: are frequencies (non-negative, sequential map); the rest are signed enrichment/attribution scores
#: (diverging map). Do not log-transform the frequencies to force a common look: that manufactures
#: negative values out of data that has none.
EXAMPLE_SOURCES = {
    'CDDM':         lambda: kdata.load('cddm'),
    'SD freq p1':   lambda: pd.read_parquet(PSSM / 'sd_freq_p1.parquet'),
    'SD freq p2':   lambda: pd.read_parquet(PSSM / 'sd_freq_p2.parquet'),
    'PSPA':         lambda: pd.read_parquet(PSSM / 'pspa_enrich.parquet'),
    'MLP-attr':     lambda: pd.read_parquet(MLP_ATTR),
    'SD enrich p1': lambda: pd.read_parquet(PSSM / 'sd_ptyr_p1.parquet'),
    'SD enrich p2': lambda: pd.read_parquet(PSSM / 'sd_ptyrvar_p2.parquet'),
    'X5-Y-X5':      lambda: pd.read_parquet(PSSM / 'sd_x5yx5_p2.parquet'),
}

#: kinases carrying all eight matrices (ABL1 and SRC are the only two; the p1 screen covers 4 kinases
#: and the X5-Y-X5 library 5, and those two sets intersect here).
EXAMPLE_KINASES = ['ABL1', 'SRC']

#: the eight matrices split by what kind of quantity they are, drawn as two figures per kinase so each
#: group keeps one colour map and one scale family. Sizing is per heatmap, so the two tile exactly.
EXAMPLE_GROUPS = {'freq':   ['CDDM', 'SD freq p1', 'SD freq p2'],
                  'signed': ['PSPA', 'MLP-attr', 'SD enrich p1', 'SD enrich p2', 'X5-Y-X5']}

#: panel title per source. p1/p2 are the two distinct published library DESIGNS, not replicates, so
#: they are labelled by design: pTyr (paper 1, eLife 35190), pTyr-Var and X5-Y-X5 (both paper 2,
#: eLife 82345). All three are surface display, so all read "SD, <design>". The freq/enrich form is
#: NOT repeated here because the group super-title already carries it, which also keeps each title to
#: one line inside the narrow columns.
EXAMPLE_TITLE = {
    'CDDM': 'CDDM', 'SD freq p1': 'SD, pTyr', 'SD freq p2': 'SD, pTyr-Var',
    'PSPA': 'PSPA', 'MLP-attr': 'MLP-attr', 'SD enrich p1': 'SD, pTyr',
    'SD enrich p2': 'SD, pTyr-Var', 'X5-Y-X5': 'SD, X5-Y-X5',
}

#: figure title suffix per group. "frequency" vs "enrichment" are the terms a reader knows; "signed"
#: is our internal word for "has negative values" and does not belong on a figure. MLP-attr is strictly
#: an attribution rather than an enrichment, so the legend, not the title, carries that precision.
EXAMPLE_GROUP_TITLE = {'freq': 'frequency', 'signed': 'enrichment'}


def plot_method_examples(kinases=EXAMPLE_KINASES, window=5, panel_w=None, height=2.0, wspace=0.10):
    """Two figures per kinase: the frequency matrices, then the signed ones. Splitting them keeps one
    colour map per figure (frequencies white-to-red, signed blue-white-red) instead of mixing both in
    one row, and lets each be placed on its own. `panel_w` is inches per heatmap (default: an eighth of
    the journal width, so all eight panels together span it), so cells are the same size in both.
    Position 0 is dropped and marked by the divider; each panel is scaled to its own range."""
    panel_w = panel_w or paper_panel(1.0)[0] / 8
    srcs = {name: loader() for name, loader in EXAMPLE_SOURCES.items()}
    positions = [p for p in range(-window, window + 1) if p != 0]
    for kinase in kinases:
        for group, names in EXAMPLE_GROUPS.items():
            panels = {}
            for name in names:
                src = srcs[name]
                if kinase not in src.index:
                    print(f'  {kinase}: no {name}, skipped')
                    continue
                m = recover_pssm(src.loc[kinase].dropna())
                panels[EXAMPLE_TITLE[name]] = m.reindex(columns=[c for c in positions if c in m.columns])
            if not panels:
                continue
            fig, _ = plot_pssm_heatmaps(panels, positions=positions, drop_zero=True,  # signed auto-detected
                                        panel=(panel_w, height), wspace=wspace,
                                        title=f'{kinase}, {EXAMPLE_GROUP_TITLE[group]}',
                                        cmap='Reds', cmap_signed='RdBu_r',
                                        title_fontsize=6.5, tick_fontsize=5)
            save_svg(FIG / f'method_example_{kinase}_{group}.svg')
            plt.close(fig)
            print(f'{kinase} [{group}]: {len(panels)} panels -> method_example_{kinase}_{group}.svg')


def plot_method_example_key(out=None, frac=1 / 3, height=0.62):
    """Small in-panel key for the g/h example rows, defining the two surface-display quantities that
    are the least self-evident. Each term sits beside its own color ramp, so a reader can read the SD
    panels without the caption. Frequency is white-to-red (0 to max); enrichment is the diverging map
    (blue depleted, white neutral, red enriched). Drop the saved SVG into the figure near g/h."""
    out = out or FIG / 'method_example_key.svg'
    grad = np.linspace(0, 1, 256)[None, :]
    rows = [('Reds',   'SD frequency',  'AA frequency among selected display peptides'),
            ('RdBu_r', 'SD enrichment', 'enrichment relative to the input library')]
    fig, axes = plt.subplots(2, 1, figsize=(paper_panel(frac)[0], height))
    for ax, (cmap, term, desc) in zip(axes, rows):
        ax.imshow(grad, aspect='auto', cmap=cmap, extent=[0, 1, 0, 1], vmin=0, vmax=1)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec='0.3', lw=0.6))   # ramp border
        ax.set_xlim(0, 12); ax.set_ylim(0, 1)
        ax.text(1.5, 0.5, f'{term}: {desc}', va='center', ha='left', fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, hspace=0.9)
    save_svg(out)
    plt.close(fig)
    print('  wrote', out)


def plot_cddm_bygroup(refmax):
    "Grouped bars — the three CDDM representations vs PSPA, by kinase group, all on the same PSPA∩CDDM set."
    cddm = ['CDDM freq', 'CDDM log-odds', 'CDDM MLP-attr']
    common = set(load_matrix('pspa').index)                         # fair: one kinase set for all 3 reps
    for src in ('cddm', 'cddm_LO', str(MLP_ATTR)):
        common &= set(load_matrix(src).index)
    ref = refmax[refmax.method.isin(cddm) & refmax.kinase.isin(common)]
    n = ref.kinase.nunique()
    g2 = kdata.load('kinase_info').drop_duplicates('kinase').set_index('kinase')['group']
    for metric in VS_METRICS:
        fname = f'compare_cddm_vs_pspa_{metric}_bygroup.svg'
        wide = ref.pivot(index='kinase', columns='method', values=metric)
        wide['group'] = wide.index.map(g2)
        wide = wide.dropna(subset=['group'])
        order = [g for g in group_color if g in set(wide['group'])]
        ax = plot_group_bar(wide, value_cols=cddm, group='group', order=order, figsize=(11, 4),
                            rotation=30, fontsize=11, palette='Set2',
                            title=f'CDDM representations vs PSPA — {METRIC_TITLE[metric]} (n={n})')
        ax.set_ylabel(f'{METRIC_TITLE[metric]} vs PSPA', fontsize=11)
        save_svg(FIG / fname)
        plt.close('all')


def plot_vs_pspa_perkinase(ref12, frac=0.62, ratio=3.05, annot_fontsize=6, order_by='spearman',
                           cbar=False, title_fontsize=8):
    """Per-kinase agreement with PSPA, one paper-sized panel per metric: methods as rows, the shared
    tyrosine kinases as columns. Columns are ordered once (by `order_by`, descending mean over the
    displayed methods) and that order is reused for every metric, so the panels stack comparably.
    No colorbar by default (cells are annotated); the color scale is per panel, so do not read color
    across metrics - Spearman and AP have different maxima."""
    keys = [k for k, _ in VS_ROWS]
    labels = [lbl for _, lbl in VS_ROWS]
    piv0 = ref12.pivot(index='method', columns='kinase', values=order_by).reindex(keys)
    kinases = piv0.mean().sort_values(ascending=False).index.tolist()
    for metric in VS_METRICS:
        piv = ref12.pivot(index='method', columns='kinase', values=metric).reindex(index=keys,
                                                                                  columns=kinases)
        fig, ax = plt.subplots(figsize=paper_panel(frac, ratio=ratio))
        sns.heatmap(piv, annot=np.vectorize(_annot)(piv.values), fmt='', annot_kws={'fontsize': annot_fontsize},
                    cmap='Reds', vmin=min(0.0, float(np.nanmin(piv.values))),
                    vmax=float(np.nanmax(piv.values)), linewidths=0.4, cbar=cbar,
                    cbar_kws={'shrink': 0.7} if cbar else None, ax=ax)
        ax.set_title(f'{METRIC_TITLE[metric]} vs PSPA', fontsize=title_fontsize, pad=3)
        ax.set_xticklabels(kinases, rotation=40, ha='right', fontsize=7)
        ax.set_yticklabels(labels, rotation=0, fontsize=7)
        ax.set_xlabel(''); ax.set_ylabel('')
        ax.tick_params(length=2, pad=1.5)
        if cbar:
            ax.collections[0].colorbar.ax.tick_params(labelsize=6, length=2, pad=1.5)
        save_svg(FIG / f'compare_vs_pspa_perkinase_{metric}.svg')
        plt.close(fig)


# ---------------------------------------------------------------- main


def main():
    if not MLP_ATTR.exists():
        sys.exit(f'{MLP_ATTR} not found - build it in motif_17 first')

    shared = shared_kinases()
    n = len(shared)
    print(f'{n} shared kinases: {shared}')

    # Fig 5 compares across PSPA, CDDM and surface display; SD carries no phospho-priming, so every
    # method is scored on the 20 standard AAs here for a fair like-for-like comparison.
    with scored_on(AA20):
        res, labels = compare(shared)                    # Fig 5 a/d (method x method)
        ref12 = vs_pspa(restrict=shared)                 # Fig 5 b/e/c/f (vs PSPA, 12 shared Tyr)
    refmax = vs_pspa()                                   # own PSPA overlap (no SD) keeps pS/pT/pY -> feeds Fig 3b
    res.to_csv(OUT / 'compare_methods.csv', index=False)
    mats = matrices(res, labels)
    for metric, M in mats.items():
        print(f'\n=== {metric} method×method (mean over {n} kinases) ===')
        print(M.round(2).to_string())
    pd.concat({m: M for m, M in mats.items()}).to_csv(OUT / 'compare_methods_matrix.csv')
    plot_matrix(mats, n)

    ref12.assign(set='shared12').to_csv(OUT / 'compare_vs_pspa.csv', index=False)
    # per-kinase agreement with PSPA over each method's own PSPA overlap (the n~293 by-group set);
    # persisted so the paper Fig 5 panels derive from it rather than re-scoring (persist-then-derive).
    refmax.assign(set='own_overlap').to_csv(OUT / 'compare_vs_pspa_bygroup.csv', index=False)
    for tag, ref in [('12 shared', ref12), ('own overlap', refmax)]:
        print(f'\n=== vs PSPA ({tag}) ===')
        print(ref.groupby('method')[VS_METRICS].mean()
              .join(ref.groupby('method')['kinase'].nunique().rename('n')).reindex(NON_PSPA).round(3).to_string())
    plot_vs_pspa_bars(ref12, refmax)
    plot_vs_pspa_perkinase(ref12)
    plot_vs_pspa_violin(ref12)                       # paper Fig 5 c/f
    plot_cddm_bygroup(refmax)
    plot_method_examples()                           # paper Fig 5 g/h
    plot_method_example_key()                        # + the in-panel SD key

    print('\nfigures:', FIG / 'compare_methods.svg', '|', FIG / 'compare_vs_pspa.svg',
          '|', FIG / 'compare_vs_pspa_perkinase.svg')


if __name__ == '__main__':
    main()
