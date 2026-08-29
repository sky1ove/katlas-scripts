"""pathway_01b · Site-centric target selection (alternative to pathway_01's top-2%).

An alternative way to turn the CDDM score matrix into each kinase's substrate set, following the
Cantley/Johnson 2023 kinome-atlas logic. Where pathway_01 takes a fixed top 2% of sites *per
kinase* (which ignores promiscuity and is not comparable across kinases), this method is
*site-centric*: a site is a kinase's substrate only when that kinase is one of the top-ranked
kinases *for that site*.

Two steps, both on the score matrix pathway_01 already wrote (no re-scoring):

  1. per-kinase percentile normalize: convert each kinase's raw scores to a 0-100 percentile across
     all sites, so kinases become comparable (the acceptor is already baked into the score, so a Ser
     site scores low for a Tyr kinase and vice versa - no explicit acceptor split needed);
  2. per-site assignment: for each site, rank the kinases by percentile and assign the site to a
     kinase only if that kinase is in the site's top TOP_K *and* clears PCT_FLOOR.

A kinase's target set is then promiscuity-aware (a sharp kinase gets few sites, a flat one more) and
every assigned site is one the kinase genuinely tops, not merely its own best 2%.

Inputs   out/pathway_cddm_score.parquet (pathway_01)
Outputs  out/pathway_cddm_sc_info.parquet   per kinase: assigned sites/genes per acceptor (+ union),
                                             schema-compatible with pathway_02 (genes, *_sites, *_cnt)

Run:  python nbs/pathway_01b_sitecentric.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from paths import OUT

import kdata

TOP_K = 3           # a site is a kinase's substrate if the kinase is in the site's top-K kinases
PCT_FLOOR = 95      # ...and the kinase's percentile for that site clears this
ACCEPTORS = ['S', 'T', 'Y']
META = ['site_seq', 'sub_site', 'sub_gene', 'acceptor']


def assign(score, kin_group):
    """Acceptor-aware boolean (site x kinase) mask. Tyrosine kinases (group TK) draw substrates from
    Y sites only, Ser/Thr kinases from S/T sites only - the score alone does not enforce this (the
    MLP barely weights the acceptor, so a Tyr kinase would otherwise 'select' mostly S/T sites).
    Within each acceptor pool a site is assigned to a kinase iff top-K for the site and >= PCT_FLOOR."""
    K = [c for c in score.columns if c not in META]
    acc = score['acceptor']
    mask = pd.DataFrame(False, index=score.index, columns=K)
    pools = [('Y', acc == 'Y', [k for k in K if kin_group.get(k) == 'TK']),
             ('S/T', acc.isin(['S', 'T']), [k for k in K if kin_group.get(k) != 'TK'])]
    for label, site_sel, kinases in pools:
        if not kinases:
            continue
        sub = score.loc[site_sel, kinases]
        pct = sub.rank(pct=True) * 100.0                  # percentile within kinase, over this pool
        site_rank = pct.rank(axis=1, ascending=False, method='min')   # rank kinases within the pool
        m = (site_rank <= TOP_K) & (pct >= PCT_FLOOR)
        mask.loc[site_sel, kinases] = m
        print(f'  {label} pool: {sub.shape[0]} sites x {len(kinases)} kinases | '
              f'assigned {int(m.values.sum()):,}')
    return mask, K


def build_info(score, mask, kinases):
    "Per-kinase assigned genes/sites, split by acceptor, schema-compatible with pathway_02."
    acc = score['acceptor'].values
    rows = {}
    for k in kinases:
        sel = mask[k].values
        out = {}
        genes_all = set()
        for a in ACCEPTORS:
            idx = sel & (acc == a)
            n = int(idx.sum())
            g = set().union(*score.loc[idx, 'sub_gene']) if n else set()
            s = set().union(*score.loc[idx, 'sub_site']) if n else set()
            out[f'{a}_cnt'] = n
            out[f'{a}_genes'] = g
            out[f'{a}_sites'] = s
            genes_all |= g
        out['genes'] = genes_all
        rows[k] = out
    info = pd.DataFrame.from_dict(rows, orient='index')
    return info


def run(method):
    "Site-centric selection for one scoring method's score matrix (cddm/pspa/mlp)."
    print(f'\n== {method.upper()} ==')
    score = pd.read_parquet(OUT / f'pathway_{method}_score.parquet')
    kin_group = kdata.load('kinase_info').drop_duplicates('kinase').set_index('kinase')['group'].to_dict()
    mask, kinases = assign(score, kin_group)
    info = build_info(score, mask, kinases)

    n = info.genes.map(len)
    print(f'  target genes per kinase: median {int(n.median())} | '
          f'range {int(n.min())}-{int(n.max())}')
    print('  per-acceptor assigned counts: median',
          {a: int(info[f"{a}_cnt"].median()) for a in ACCEPTORS})

    info.to_parquet(OUT / f'pathway_{method}_sc_info.parquet')
    print(f'  {len(info)} kinases -> {OUT / f"pathway_{method}_sc_info.parquet"}')


def main():
    methods = sys.argv[1:] or ['cddm', 'pspa', 'mlp']
    for m in methods:
        run(m)


if __name__ == '__main__':
    main()
