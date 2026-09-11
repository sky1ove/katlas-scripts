"""pathway_04 · Curated CDDM example panel: one kinase group per panel.

The showcase figure: for a well-studied kinase from each kinase group, the pathways its CDDM-
predicted substrates fall into (site-centric selection, local ORA vs the phosphoproteome background,
specificity ranking) are that kinase's known biology.

Method matches the chosen pipeline (pathway_07): site-centric target sets, hypergeometric ORA
against the phosphoproteome, pathways ranked by specificity (enrichment minus the kinome mean).
Every bar is a pathway the enrichment found (the kinase's most specific predicted-substrate
pathways); red marks those that are also in the kinase's Reactome annotation (a recovered known
pathway), gray those predicted by the enrichment but not annotated to that kinase in Reactome (often
its real biology Reactome does not annotate, e.g. CAMK2A -> Neurexins / synapse, not a wrong hit).

Groups shown: one well-studied kinase from each of eight kinase groups (AGC, Atypical, CAMK, CMGC,
STE, Other, TK, TKL), matching the eight panels of the paper figure. CK1 (acidic) and the singleton
NEK are the most promiscuous motifs, so their predicted substrates hit generic pathways; that is
noted in the text rather than shown as empty panels.

Inputs   out/pathway_cddm_sc_info.parquet (pathway_01b), kdata: human_site, kinase_info;
         Data.reactome_pathway  (all via pathway_07.universe / ora_matrix)
Outputs  fig/pathway_cddm_examples_col1.svg, fig/pathway_cddm_examples_col2.svg (two single-column
         figures, placed independently), fig/pathway_cddm_examples_legend.svg (separate color key),
         out/pathway_cddm_examples_stats.md

Run:  python nbs/pathway_04_cddm_examples.py   (needs pathway_01b)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from paths import FIG, OUT

import kdata
from kplot.utils import paper_panel, save_svg
from pathway_07_local_ora import ora_matrix, universe

# one well-studied kinase per group whose predicted-substrate pathways recover cleanly
EXAMPLES = [('AGC', 'AKT1'), ('Atypical', 'ATM'), ('CAMK', 'CAMK2A'), ('CMGC', 'ERK2'),
            ('STE', 'PAK4'), ('Other', 'BUB1'), ('TK', 'SYK'), ('TKL', 'TAK1')]
TAG = 'cddm_sc'
TOP_N = 7
MAX_LABEL = 38          # over this, truncate at the last whole word (never mid-word)
COL_W_MM = 90           # width of each single-column figure; extra width goes to the label column,
                        # not the bars (widen this together with MAX_LABEL to keep bar width constant)
RATIO = 1.4             # column height = 180 mm / RATIO (4 panels stacked)
HIT, MISS = '#d62728', '#bdbdbd'


def clip(name):
    "Full Reactome name, truncated at the last whole word so nothing is cut mid-word."
    if len(name) <= MAX_LABEL:
        return name
    return name[:MAX_LABEL].rsplit(' ', 1)[0].rstrip(' ,') + '…'


def plot_kinase(ax, kinase, group, spec, names, ref_ids, show_xlabel):
    # take the top pathways, but skip any whose shortened label repeats one already shown (Reactome
    # has many near-duplicate variants, e.g. ATM's HR-repair pathways, that truncate to the same text)
    picked, seen = [], set()
    for r in spec.sort_values(ascending=False).index:
        lab = clip(names.get(r, r))
        if lab in seen:
            continue
        seen.add(lab); picked.append((r, lab))
        if len(picked) == TOP_N:
            break
    picked = picked[::-1]
    rids = [r for r, _ in picked]; labels = [lab for _, lab in picked]
    colors = [HIT if r in ref_ids else MISS for r in rids]
    ax.barh(range(len(rids)), spec[rids].values, color=colors, height=0.72)
    ax.set_yticks(range(len(rids))); ax.set_yticklabels(labels)
    ax.set_title(f'{kinase}  ({group})', loc='left', fontweight='bold', pad=2)
    if show_xlabel:
        ax.set_xlabel('Pathway specificity\n(Δ –log₁₀ p vs kinome mean)', labelpad=1)
    ax.tick_params(length=2, pad=1.5); ax.margins(x=0.02)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def build_column(subset, spec, names, ann, kin_uni, out_svg):
    "One single-column figure (4 panels stacked), saved separately so columns can be placed freely."
    paper_panel(1)                                            # 7 pt fonts / 0.6 pt lines
    fig, axes = plt.subplots(len(subset), 1, figsize=(COL_W_MM / 25.4, 180 / RATIO / 25.4))
    stats = []
    for i, ((group, kinase), ax) in enumerate(zip(subset, axes)):
        ref_ids = ann.get(kin_uni.get(kinase), set()) & set(spec.index)
        plot_kinase(ax, kinase, group, spec[kinase], names, ref_ids,
                    show_xlabel=(i == len(subset) - 1))
        top = spec[kinase].sort_values(ascending=False).head(TOP_N)
        stats.append(f'- **{kinase}** ({group}): ' +
                     '; '.join(names.get(r, r) + (' [ann]' if r in ref_ids else '')
                               for r in top.index[:4]))
        print(stats[-1][:150])
    fig.tight_layout()
    save_svg(out_svg)
    plt.close('all')
    print('  wrote', out_svg)
    return stats


def main():
    bg, N, pw, pw_u, pw_K, names, ann = universe()
    kin_uni = kdata.load('kinase_info').drop_duplicates('kinase').set_index('kinase')['uniprot'].to_dict()
    M = ora_matrix(TAG, bg, N, pw, pw_u, pw_K)
    spec = M.sub(M.mean(axis=1), axis=0)

    missing = [k for _, k in EXAMPLES if k not in spec.columns]
    if missing:
        sys.exit(f'not scored: {missing}')

    # two single-column figures, preserving the original left/right column assignment
    left, right = EXAMPLES[0::2], EXAMPLES[1::2]
    stats = build_column(left, spec, names, ann, kin_uni, FIG / 'pathway_cddm_examples_col1.svg')
    stats += build_column(right, spec, names, ann, kin_uni, FIG / 'pathway_cddm_examples_col2.svg')

    save_legend(FIG / 'pathway_cddm_examples_legend.svg')     # separate box, place it freely
    (OUT / 'pathway_cddm_examples_stats.md').write_text('\n'.join(stats) + '\n')


def save_legend(out_svg):
    "The color key as its own tightly-cropped SVG, so it can be placed anywhere in the layout."
    paper_panel(1)
    fig = plt.figure(figsize=(80 / 25.4, 6 / 25.4))
    fig.legend(handles=[Patch(color=HIT, label='Enriched & annotated in Reactome'),
                        Patch(color=MISS, label='Enriched, not currently annotated in Reactome')],
               loc='center', ncol=2, frameon=False)
    save_svg(out_svg)
    plt.close('all')
    print('  wrote', out_svg)


if __name__ == '__main__':
    main()
