"""motif_23 · Rationale panels (a-d) for the EGFR/EPHA3 structural figure (Fig 6).

The left column motivates the structural analysis: real tyrosine-acceptor sites carry little flanking
phospho-priming, yet the peptide array (PSPA) has most TK kinases preferring a flanking phospho-tyrosine
(pY) at -1/+1/+2. So the array's pY preference is an intrinsic binding preference whose in-cell
relevance is unclear, which the structural panels (e-g) then explain.

Panels (each sized for 1/3 of the 180 mm figure width via `kplot.utils.paper_panel`):
  a. **Y-site sequence logo** (-5..+5), observed `ks_dataset` Y-acceptor sites. Two versions: a
     **frequency** logo (letter height = raw frequency) and an **IC** logo (information content, near
     flat = weak motif). Pick one.
  b. **sty frequency** — flanking pS/pT/pY frequency by position among the same Y-sites (a minority).
  c. **PSPA TK preference by position** — # TK kinases (of 78) whose rank-1 array residue is pY vs pS/T
     (pS==pT array duplicate dropped), by position. WHERE the preference sits (-1/+1/+2).
  d. **Overall** — of the 78 TK kinases, how many have pY as the #1 residue at −1/+1/+2 (each kinase
     once): 66/78 = 85% (definition B). HOW MANY of the family, the one-number summary (complements c,
     does not repeat it); 'other' = top residue at all of −1/+1/+2 is a regular amino acid.

Also emits the alternative TK views (value, top-5) for choosing. Every number computed live.

Inputs   kdata: pspa_scale, kinase_info, ks_dataset(thr=40)
Outputs  out/priming_rationale.csv;
         fig/panel_a_logo_{freq,IC}.svg, fig/panel_b_sty_freq.svg, fig/panel_c_tk_rank1.svg,
         fig/panel_d_overall.svg, fig/panel_d_overall_bars.svg,
         fig/alt_tk_{value,top5}.svg
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paths import FIG, OUT

import kdata
import katlas.plot as kp
from katlas.pssm import get_prob
from katlas.utils import sty_color
from kplot.utils import paper_panel, save_svg, set_sns

FLANK = [p for p in range(-5, 6) if p != 0]
KEY = [-1, 1, 2]                                    # the positions the structural panels (e-g) address
C = 20
AA20 = list('ACDEFGHIKLMNPQRSTVWY')


def parse(c):
    g = re.fullmatch(r'(-?\d+)([A-Za-z])', str(c))
    return (int(g.group(1)), g.group(2)) if g else (None, None)


def tk_kinases():
    ki = kdata.load('kinase_info').drop_duplicates('kinase').set_index('kinase')
    return set(ki.index[ki.group == 'TK'])


def ysites():
    "Unique tyrosine-acceptor sites: dedup on the uppercase window so each site is counted once "
    "(not once per kinase), then keep Y-acceptor windows."
    ks = kdata.ks_dataset(thr=40)
    ks = ks.assign(u=ks.site_seq.str.upper()).drop_duplicates('u')
    return ks[ks.site_seq.str[C].isin(['Y', 'y'])]


# ---------- data ----------

def sty_freq(ys):
    "Per flank position, fraction of Y-acceptor windows with a flanking pS / pT / pY."
    seqs = [s for s in ys.site_seq if isinstance(s, str) and len(s) == 2 * C + 1]
    arr = np.array([list(s) for s in seqs])
    return len(seqs), {r: {p: float(np.mean(arr[:, C + p] == r)) for p in FLANK} for r in ('s', 't', 'y')}


def tk_pref_stats():
    "Per position: mean scaled pY/pS-T value, and rank-1 / top-5 kinase counts. Plus family partition."
    TK = tk_kinases()
    p = kdata.load('pspa_scale')
    p = p[~p.index.str.contains('_TYR')]
    tk = [k for k in p.index if k in TK]
    M = p.loc[tk]
    flank_cells = [c for c in M.columns
                   if parse(c)[0] not in (None, 0) and abs(parse(c)[0]) <= 5 and parse(c)[1] != 's']
    d = dict(py_val={}, pst_val={}, py_r1={p: 0 for p in FLANK}, pst_r1={p: 0 for p in FLANK},
             py_t5={p: 0 for p in FLANK}, pst_t5={p: 0 for p in FLANK})
    for pos in FLANK:
        d['py_val'][pos] = float(M.get(f'{pos}y', pd.Series(np.nan)).mean())
        d['pst_val'][pos] = float(M.get(f'{pos}t', pd.Series(np.nan)).mean())
    # family partition (each kinase once) uses definition B: the rank-1 residue at -1/+1/+2 only.
    # 'other' = the kinase's top residue at all of -1/+1/+2 is a regular amino acid (no flanking priming).
    partition = {'pY': 0, 'pS/T': 0, 'other': 0}
    for k in tk:
        row = M.loc[k]
        tops = {}
        for pos in FLANK:
            cells = {r: row.get(f'{pos}{r}', np.nan) for r in AA20 + ['t', 'y']}
            cells = {r: v for r, v in cells.items() if pd.notna(v)}
            top = max(cells, key=cells.get) if cells else None
            tops[pos] = top
            if top == 'y':
                d['py_r1'][pos] += 1
            elif top == 't':
                d['pst_r1'][pos] += 1
        for c in row[flank_cells].dropna().sort_values(ascending=False).head(5).index:
            pos, res = parse(c)
            if res == 'y':
                d['py_t5'][pos] += 1
            elif res == 't':
                d['pst_t5'][pos] += 1
        key_tops = [tops[pos] for pos in KEY]
        partition['pY' if 'y' in key_tops else ('pS/T' if 't' in key_tops else 'other')] += 1
    return len(tk), d, partition


# ---------- panel a: logos ----------

def _ysite_pssm(ys):
    P = get_prob(ys, 'site_seq')
    return P[[c for c in P.columns if isinstance(c, int) and -5 <= c <= 5]]


def panel_a_logos(ys):
    sub = _ysite_pssm(ys)
    for kind, ytitle, fn in [('freq', 'frequency', 'panel_a_logo_freq.svg'),
                             ('IC', 'IC (bits)', 'panel_a_logo_IC.svg')]:
        paper_panel(1 / 3, ratio=2.3)                         # wide, short logo strip
        fig, ax = plt.subplots(figsize=(2.36, 1.05))
        if kind == 'freq':
            kp.plot_logo_raw(sub, ax=ax, title='', ytitle=ytitle)
        else:
            kp.plot_logo(sub, ax=ax, title='')
            ax.set_ylabel(ytitle)
        ax.set_xlabel('position')
        save_svg(FIG / fn)
        plt.close(fig)


# ---------- shared grouped-bar layout (panels b, c and alternatives) ----------

def _grouped(series, colors, labels, ylabel, fname, ratio=1.35, pct=False, annotate_idx=None):
    paper_panel(1 / 3, ratio=ratio)
    fig, ax = plt.subplots(figsize=paper_panel(1 / 3, ratio=ratio))
    x = np.array(FLANK)
    n = len(series)
    w = .8 / n
    for i, (s, col, lab) in enumerate(zip(series, colors, labels)):
        off = (i - (n - 1) / 2) * w
        vals = [s[p] * (100 if pct else 1) for p in FLANK]
        ax.bar(x + off, vals, w, color=col, label=lab)
        if annotate_idx == i and not pct:
            for p in FLANK:
                if s[p]:
                    ax.annotate(str(s[p]), (p + off, s[p]), xytext=(0, 1.2), textcoords='offset points',
                                ha='center', fontsize=5, color='#7f3b08')
    ax.set_xticks(x, [str(p) for p in FLANK], fontsize=6)
    ax.set_xlabel('position')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=5.5, frameon=False, handlelength=1, borderpad=.1, labelspacing=.25)
    ax.spines[['top', 'right']].set_visible(False)
    save_svg(FIG / fname)
    plt.close(fig)


def panel_b_sty(freq):
    _grouped([freq['s'], freq['t'], freq['y']],
             [sty_color['s'], sty_color['t'], sty_color['y']], ['pS', 'pT', 'pY'],
             'flank phospho\n(% of Y-sites)', 'panel_b_sty_freq.svg', ratio=1.65, pct=True)


def panel_c_rank1(d, n_tk):
    _grouped([d['py_r1'], d['pst_r1']], [sty_color['y'], sty_color['s']], ['pY #1', 'pS/T #1'],
             f'TK kinases (of {n_tk})', 'panel_c_tk_rank1.svg', ratio=1.65, annotate_idx=0)


def alt_tk_value(d, n_tk):
    _grouped([d['py_val'], d['pst_val']], [sty_color['y'], sty_color['s']], ['pY', 'pS/T'],
             'mean pref\n(% of position)', 'alt_tk_value.svg', pct=True)


def alt_tk_top5(d, n_tk):
    _grouped([d['py_t5'], d['pst_t5']], [sty_color['y'], sty_color['s']], ['pY', 'pS/T'],
             f'top-5 for N kinases\n(of {n_tk})', 'alt_tk_top5.svg', annotate_idx=0)


# ---------- panel d: overall family summary ----------

def panel_d_stacked(partition, n_tk):
    "One horizontal stacked bar: the 78 TK kinases split by their flanking-phospho preference."
    order = ['pY', 'pS/T', 'other']
    cols = {'pY': sty_color['y'], 'pS/T': sty_color['s'], 'other': '#c9c9c9'}
    fig, ax = plt.subplots(figsize=paper_panel(1 / 3, ratio=2.6))
    left = 0
    for key in order:
        v = partition[key]
        ax.barh(0, v, left=left, color=cols[key], edgecolor='white', lw=.5, label=f'{key} ({v})')
        if v >= 10:                                            # wide segment: label inside
            ax.text(left + v / 2, 0, str(v), ha='center', va='center', fontsize=7, color='white')
        left += v
    ax.set_xlim(0, n_tk)
    ax.set_ylim(-.5, .9)
    ax.set_yticks([])
    pct = round(partition['pY'] / n_tk * 100)
    ax.set_xlabel(f'{partition["pY"]}/{n_tk} ({pct}%) TK kinases: pY is #1 at −1/+1/+2')
    ax.legend(ncol=3, fontsize=5.5, frameon=False, loc='lower center',
              bbox_to_anchor=(.5, .70), handlelength=1, columnspacing=1, handletextpad=.4)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    save_svg(FIG / 'panel_d_overall.svg')
    plt.close(fig)


def panel_d_bars(partition, n_tk):
    "Alternative: three vertical bars for the family partition."
    fig, ax = plt.subplots(figsize=paper_panel(1 / 3, ratio=1.35))
    keys = ['pY', 'pS/T', 'other']
    cols = [sty_color['y'], sty_color['s'], '#c9c9c9']
    ax.bar(keys, [partition[k] for k in keys], color=cols, width=.65)
    for i, k in enumerate(keys):
        ax.annotate(str(partition[k]), (i, partition[k]), xytext=(0, 1.5),
                    textcoords='offset points', ha='center', fontsize=6.5)
    ax.set_ylabel(f'TK kinases (of {n_tk})')
    ax.set_title('#1 residue at −1/+1/+2 (each kinase once)', fontsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    save_svg(FIG / 'panel_d_overall_bars.svg')
    plt.close(fig)


def main():
    set_sns()
    ys = ysites()
    n_y, freq = sty_freq(ys)
    n_tk, d, partition = tk_pref_stats()

    pd.DataFrame({
        'position': FLANK,
        'ks_ysite_pS_freq': [freq['s'][p] for p in FLANK],
        'ks_ysite_pT_freq': [freq['t'][p] for p in FLANK],
        'ks_ysite_pY_freq': [freq['y'][p] for p in FLANK],
        'tk_pY_mean_scaled': [d['py_val'][p] for p in FLANK],
        'tk_pST_mean_scaled': [d['pst_val'][p] for p in FLANK],
        'tk_n_pY_rank1': [d['py_r1'][p] for p in FLANK],
        'tk_n_pST_rank1': [d['pst_r1'][p] for p in FLANK],
        'tk_n_pY_top5': [d['py_t5'][p] for p in FLANK],
        'tk_n_pST_top5': [d['pst_t5'][p] for p in FLANK],
    }).round(4).to_csv(OUT / 'priming_rationale.csv', index=False)
    print(f'Y-sites n={n_y} | TK n={n_tk} | family partition {partition}')

    panel_a_logos(ys)
    panel_b_sty(freq)
    panel_c_rank1(d, n_tk)
    panel_d_stacked(partition, n_tk)
    panel_d_bars(partition, n_tk)
    alt_tk_value(d, n_tk)
    alt_tk_top5(d, n_tk)
    print('wrote panels a-d (+ alternatives) to', FIG)


if __name__ == '__main__':
    main()
