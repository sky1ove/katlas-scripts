"""pathway_01 · Score the human phosphoproteome and pick each kinase's predicted targets.

Step 1 of the pathway analysis: turn each kinase's PSSM into a concrete target gene set, so the
next script can ask which Reactome pathways those targets fall in.

The hard part is choosing *how many* sites count as a kinase's targets, per phosphoacceptor. A
fixed score cutoff does not transfer between kinases, and a fixed count ignores that a Ser/Thr
kinase's Thr sites should be far fewer than its Ser sites. The procedure:

  1. find the kinase's dominant acceptor from its central S/T/Y ratio;
  2. take the top TOP_PCT of sites for that acceptor, and record how many that is;
  3. scale that count to the other acceptors by the same S/T/Y ratio, and read off the score
     threshold each scaled count implies.

So the number of predicted targets is set once, on the acceptor the kinase actually prefers, and
the remaining acceptors inherit it in proportion.

Sites are deduplicated on `site_seq` first — the same 41-mer appearing at several positions would
otherwise contribute its score several times. Done for both PSPA and CDDM.

Inputs   kdata: human_site, pspa_scale, cddm
Outputs  out/pathway_{pspa,cddm}_score.parquet   scored sites (one row per unique site_seq)
         out/pathway_{pspa,cddm}_info.parquet    per kinase: thresholds, counts, target genes/sites

Run:  python nbs/pathway_01_score_sites.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from paths import OUT

import kdata
from katlas.scoring import Params, predict_kinase_df

TOP_PCT = 0.98      # quantile defining "a target" for the dominant acceptor (top 2%)
TYR_ONLY = 0.8      # central 0y at or above this -> a pure tyrosine kinase
ST_ONLY = 0.1       # central 0y at or below this -> a pure Ser/Thr kinase

ACCEPTORS = ['S', 'T', 'Y']
CNT_COLS = [f'{a}_cnt' for a in ACCEPTORS]
THR_COLS = [f'{a}_thr' for a in ACCEPTORS]
GENE_COLS = [f'{a}_genes' for a in ACCEPTORS]


def load_sites():
    "Human phosphoproteome, one row per unique site sequence, with its genes and sites collapsed."
    df = kdata.load('human_site')
    df['sub_gene'] = df.substrate_genes.str.split(' ').str[0]
    print(f'human_site: {len(df):,} rows | duplicated site_seq: {int(df.site_seq.duplicated().sum()):,}')

    uniq = df.groupby('site_seq').agg({'sub_site': lambda x: set(x.dropna()),
                                       'sub_gene': lambda x: set(x.dropna())}).reset_index()
    uniq['acceptor'] = uniq.site_seq.str[20].str.upper()
    print(f'unique site sequences: {len(uniq):,} |', dict(uniq.acceptor.value_counts()))
    return uniq


def score(sites, params_name):
    "Score every site against every kinase of one PSSM family."
    out = predict_kinase_df(sites, seq_col='site_seq', **Params(params_name))
    return pd.concat([sites, out], axis=1)


def sty_ratio(series):
    "Normalised S/T/Y ratio for one kinase, from its central 0s/0t/0y values."
    s, t, y = series
    if y >= TYR_ONLY:
        return pd.Series({'S': 0, 'T': 0, 'Y': 1})
    if y > ST_ONLY:                                   # mixed S/T/Y - only occurs in CDDM
        total = s + t + y
        return pd.Series({'S': s / total, 'T': t / total, 'Y': y / total})
    total = s + t
    return pd.Series({'S': s / total if total else 0, 'T': t / total if total else 0, 'Y': 0})


def thresholds_and_counts(series, scored, kinase):
    """Target count and score threshold per acceptor for one kinase.

    The dominant acceptor gets the top-(1-TOP_PCT) sites; the others get a count scaled by the
    kinase's S/T/Y ratio, and the threshold that count implies.
    """
    top = series.max_acceptor
    scores = scored[scored.acceptor == top][kinase]
    thr = scores.quantile(TOP_PCT)
    count = int(scores[scores >= thr].count())
    out = {f'{top}_thr': thr, f'{top}_cnt': count}

    for acceptor in series.acceptors:
        if acceptor == top:
            continue
        n = int((count / series[top]) * series[acceptor])
        out[f'{acceptor}_cnt'] = n
        out[f'{acceptor}_thr'] = scored[scored.acceptor == acceptor][kinase].nlargest(n).min()
    return pd.Series(out)


def target_sets(series, scored, kinase):
    "The genes and sites behind each acceptor's target count."
    out = {}
    for acceptor in ACCEPTORS:
        n = int(series[f'{acceptor}_cnt'])
        if n:
            top = scored[scored.acceptor == acceptor].nlargest(n, columns=kinase)
            out[f'{acceptor}_genes'] = set().union(*top.sub_gene)
            out[f'{acceptor}_sites'] = set().union(*top.sub_site)
        else:
            out[f'{acceptor}_genes'] = set()
            out[f'{acceptor}_sites'] = set()
    return pd.Series(out)


def build_info(pssms, scored):
    "Per-kinase thresholds, counts and target sets."
    ratio = pssms[['0s', '0t', '0y']].apply(sty_ratio, axis=1)
    ratio['acceptors'] = ratio.apply(lambda r: r[r != 0].index.tolist(), axis=1)
    ratio['max_acceptor'] = ratio.iloc[:, :3].idxmax(axis=1)
    print('  dominant acceptor:', dict(ratio.max_acceptor.value_counts()))

    info = ratio.apply(lambda r: thresholds_and_counts(r, scored, r.name), axis=1)
    # an acceptor no kinase in this set prefers never produces its columns at all, so pin the
    # schema rather than relying on the kinase mix (a Tyr-free subset used to KeyError here)
    info = info.reindex(columns=CNT_COLS + THR_COLS).fillna(0)
    info[CNT_COLS] = info[CNT_COLS].astype(int)

    genes = info.apply(lambda r: target_sets(r, scored, r.name), axis=1)
    genes['genes'] = genes[GENE_COLS].apply(lambda r: set().union(*r), axis=1)

    info = pd.concat([info, genes], axis=1)
    print('  target genes per kinase: median',
          int(info.genes.map(len).median()), '| range',
          f'{int(info.genes.map(len).min())}-{int(info.genes.map(len).max())}')
    return info


def main():
    OUT.mkdir(exist_ok=True)
    sites = load_sites()

    for method, params_name, pssm_name in [('pspa', 'PSPA', 'pspa_scale'),
                                           ('cddm', 'CDDM', 'cddm')]:
        print(f'\n== {method.upper()} ==')
        scored = score(sites, params_name)
        scored.to_parquet(OUT / f'pathway_{method}_score.parquet')
        print(f'  scored {scored.shape} -> {OUT / f"pathway_{method}_score.parquet"}')

        info = build_info(kdata.load(pssm_name), scored)
        info.to_parquet(OUT / f'pathway_{method}_info.parquet')
        print(f'  {len(info)} kinases -> {OUT / f"pathway_{method}_info.parquet"}')


if __name__ == '__main__':
    main()
