"""Shared helpers for the motif_* scripts (dendrogram colouring + labels)."""
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from katlas.pssm import pssm_to_seq, recover_pssm
from katlas.utils import group_color

# one consensus position: a bracket group (optionally starred) or a single char (optionally starred)
_POS_TOKEN = re.compile(r'\[[^\]]*\]\*?|[^\[\*]\*?')

#: same width as a space in a monospace font, but SVG/browsers won't collapse runs of it,
#: so the column alignment survives saving
NBSP = ' '


def kinase_group_map(info):
    "kinase -> group lookup that also matches suffixed names like `SRC_TYR`."
    k2g = info.set_index('kinase')['group'].to_dict()
    return lambda k: k2g.get(k) or k2g.get(k.split('_')[0])


def get_aligned_labels(pssms, count_map=None, thr=0.2):
    "Dendrogram labels with every motif position in a fixed-width column, so the central s/t/y* lines up."
    seqs = {i: _POS_TOKEN.findall(pssm_to_seq(recover_pssm(r), thr=thr)) for i, r in pssms.iterrows()}
    widths = [max(len(toks[c]) for toks in seqs.values()) for c in range(max(map(len, seqs.values())))]
    pref = (lambda k: f'{k} (n={count_map[k]:,})') if count_map is not None else (lambda k: f'{k}')
    pw = max(len(pref(k)) for k in seqs)
    labels = [(pref(k) + ': ').ljust(pw + 2) + ''.join(t.center(w) for t, w in zip(seqs[k], widths))
              for k in pssms.index]
    return [s.replace(' ', NBSP) for s in labels]


def get_group_colors(Z, index, group_of, labels=None, palette=group_color, default='#bdbdbd'):
    "Colour dendrogram branches/leaves by kinase group; a cluster mixing groups gets the default gray."
    to_hex = lambda g: mcolors.to_hex(palette[g]) if g in palette else default
    keys = [group_of(k) or '__NA__' for k in index]     # leaf -> group (sentinel if unknown)
    sets = [{g} for g in keys]                          # descendant-group set per node, built bottom-up
    for left, right, *_ in Z:
        sets.append(sets[int(left)] | sets[int(right)])

    def link_color_func(node_id):
        g = sets[node_id]
        if len(g) == 1:
            only = next(iter(g))
            return to_hex(only) if only != '__NA__' else default
        return default

    label_colors = {(labels[i] if labels is not None else k): to_hex(keys[i])
                    for i, k in enumerate(index)}
    return link_color_func, label_colors


def style_yticklabels(label_colors=None, monospace=False, ax=None):
    "Colour dendrogram leaf labels by a {label: colour} map, optionally forcing a monospace font."
    ax = ax or plt.gca()
    for lbl in ax.get_ymajorticklabels():
        if monospace:
            lbl.set_fontfamily('monospace')
        if label_colors:
            c = label_colors.get(lbl.get_text())
            if c:
                lbl.set_color(c)


def add_group_legend(groups, palette=group_color, ax=None):
    "Add a kinase-group colour legend in the (empty) top-left corner."
    ax = ax or plt.gca()
    handles = [Patch(color=mcolors.to_hex(palette[g]), label=g) for g in groups if g in palette]
    ax.legend(handles=handles, title='Kinase group', bbox_to_anchor=(0, 1),
              loc='upper left', fontsize=7, title_fontsize=8, frameon=False)
