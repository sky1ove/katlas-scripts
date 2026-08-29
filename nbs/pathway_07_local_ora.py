"""pathway_07 · Fast local ORA + kinome-wide method evaluation (no Reactome web API).

Everything pathway_02/02b + pathway_05 do via the Reactome web service, done locally in seconds.

Why local. The web-API sweep is ~25 min per selection, almost all of it the courtesy delay between
calls, and it enriches against a whole-genome background - wrong for kinase substrates, which are all
phosphoproteins. Here we run the over-representation test ourselves (hypergeometric) against the
PHOSPHOPROTEOME background (every protein we scored), over all human Reactome pathways, straight from
the local reactome_pathway table. ~20 s for the whole kinome, correct background, reproducible, and a
single shared pathway universe across every method.

For each scoring method (CDDM, PSPA, MLP) x selection (top-2%, site-centric) it builds the
pathway x kinase -log10(p) matrix, then scores how well the enrichment recovers each kinase's own
Reactome-annotated pathways: AUROC and Average Precision per kinase, averaged over all kinases with
>= MIN_REF annotations, with a permutation null (each kinase scored against a random kinase's
profile) so the reported kinase-specific signal (real - null) is not the generic-pathway baseline.
Raw vs specificity ranking, TK vs non-TK.

Inputs   out/pathway_{cddm,pspa,mlp}_info.parquet (top-2%, pathway_01 / pathway_06 + borrowed ratio),
         out/pathway_{cddm,pspa,mlp}_sc_info.parquet (site-centric, pathway_01b),
         kdata: human_site, kinase_info; Data.reactome_pathway
Outputs  out/pathway_localora_eval.parquet   the metric table

Run:  python nbs/pathway_07_local_ora.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from paths import OUT
from scipy.stats import hypergeom
from sklearn.metrics import average_precision_score, roc_auc_score

import kdata
from katlas.data import Data

MIN_REF = 10
N_PERM = 50
SEED = 0
PW_MIN, PW_MAX = 5, 600     # pathway size range (in the phosphoproteome background)
# pure non-biology artifacts (never a kinase function): viral / pathogen / infection pathways enrich
# via phospho-dense host proteins. Safe to drop - unlike GTPase/SUMO, which ARE real functions for
# some kinases (PAK4 -> Rho GTPase cycle, HIPK2 -> SUMOylation), so those are left to the specificity
# ranking, not hard-excluded.
ARTIFACT = ['Dengue', 'SARS-CoV', 'HIV', 'Influenza', 'Listeria', 'Virus', 'viral', 'Infecti',
            'Epstein', 'anthrax', 'toxin', 'tuberculosis', 'Leishmania', 'Mtb', 'Mycobact',
            'Legionella', 'bacteri']
ACC = ['S', 'T', 'Y']
# scoring method x selection -> info file tag
TAGS = [('CDDM', 'top2%', 'cddm'), ('CDDM', 'sitecentric', 'cddm_sc'),
        ('PSPA', 'top2%', 'pspa'), ('PSPA', 'sitecentric', 'pspa_sc'),
        ('MLP', 'top2%', 'mlp'), ('MLP', 'sitecentric', 'mlp_sc')]


def universe():
    "Phosphoproteome bg, pathway->uniprots, pathway names, and kinase->annotated pathways."
    hs = kdata.load('human_site')
    bg = set(hs.substrate_uniprot.dropna().unique())
    rp = Data.reactome_pathway()
    rp = rp[rp.species.str.contains('Homo sapiens', na=False)]
    p2u = rp.groupby('reactome_id').agg(name=('pathway', 'first'),
                                        u=('uniprot', lambda s: set(s) & bg))
    p2u = p2u[p2u.u.map(len).between(PW_MIN, PW_MAX)]
    art = p2u['name'].str.contains('|'.join(ARTIFACT), case=False, na=False)
    p2u = p2u[~art]                                       # drop pure viral/pathogen artifacts
    pw = list(p2u.index)
    pw_u = [p2u.loc[r, 'u'] for r in pw]
    pw_K = np.array([len(u) for u in pw_u])
    names = p2u['name'].to_dict()
    ann = rp[rp.reactome_id.isin(set(pw))].groupby('uniprot')['reactome_id'].agg(set)
    return bg, len(bg), pw, pw_u, pw_K, names, ann


def target_uniprots(info, bg):
    "Per-kinase set of target uniprots, from the selection's assigned sites."
    out = {}
    for k, r in info.iterrows():
        sites = set().union(*[r[f'{a}_sites'] for a in ACC])
        q = {s.rsplit('_', 1)[0] for s in sites} & bg
        if q:
            out[k] = q
    return out


def ora_matrix(tag, bg, N, pw, pw_u, pw_K):
    "pathway x kinase -log10(p) matrix by hypergeometric ORA against the phosphoproteome."
    info = pd.read_parquet(OUT / f'pathway_{tag}_info.parquet')
    tgt = target_uniprots(info, bg)
    cols = {}
    for k, q in tgt.items():
        n = len(q)
        x = np.array([len(q & u) for u in pw_u])
        nlp = -hypergeom.logsf(x - 1, N, pw_K, n) / np.log(10)
        nlp[x < 1] = 0.0
        cols[k] = nlp
    return pd.DataFrame(cols, index=pw)


def evaluate(M, ann, kin_uni, group, spec):
    "AUROC/AP per kinase (own pathways vs rest) + permutation null; overall and TK/non-TK."
    S = M.sub(M.mean(axis=1), axis=0) if spec else M
    idx = M.index
    y = {}
    keep = []
    for k in M.columns:
        u = kin_uni.get(k)
        refs = ann.get(u, set()) & set(idx) if u else set()
        if len(refs) >= MIN_REF:
            y[k] = idx.isin(refs).astype(int)
            keep.append(k)
    au = {k: roc_auc_score(y[k], S[k].values) for k in keep}
    ap = {k: average_precision_score(y[k], S[k].values) for k in keep}

    rng = np.random.default_rng(SEED)
    nulls = []
    for _ in range(N_PERM):
        perm = list(keep); rng.shuffle(perm)
        nulls.append(np.mean([roc_auc_score(y[k], S[kp].values) for k, kp in zip(keep, perm)]))
    null = float(np.mean(nulls))

    rows = {}
    for label, sel in [('all', keep),
                       ('nonTK', [k for k in keep if group.get(k) != 'TK']),
                       ('TK', [k for k in keep if group.get(k) == 'TK'])]:
        if sel:
            rows[label] = {'n_kinase': len(sel),
                           'auroc': np.mean([au[k] for k in sel]),
                           'ap': np.mean([ap[k] for k in sel])}
    rows['all']['auroc_null'] = null
    rows['all']['signal'] = rows['all']['auroc'] - null
    return rows


def main():
    bg, N, pw, pw_u, pw_K, names, ann = universe()
    info = kdata.load('kinase_info').drop_duplicates('kinase')
    kin_uni = info.set_index('kinase')['uniprot'].to_dict()
    group = info.set_index('kinase')['group'].to_dict()
    print(f'universe: {N} bg proteins, {len(pw)} pathways')

    records = []
    for method, selection, tag in TAGS:
        if not (OUT / f'pathway_{tag}_info.parquet').exists():
            print(f'skip {tag}: no info file'); continue
        M = ora_matrix(tag, bg, N, pw, pw_u, pw_K)
        for spec in (False, True):
            res = evaluate(M, ann, kin_uni, group, spec)
            for split, v in res.items():
                records.append({'method': method, 'selection': selection,
                                'ranking': 'specificity' if spec else 'raw', 'split': split, **v})
        print(f'  {method:5s} {selection:11s}: {M.shape[1]} kinases scored')

    tab = pd.DataFrame(records)
    tab.to_parquet(OUT / 'pathway_localora_eval.parquet')

    show = tab[tab.split == 'all'][['method', 'selection', 'ranking', 'n_kinase',
                                    'auroc', 'auroc_null', 'signal', 'ap']]
    print('\nLocal-ORA (phosphoproteome background) kinome-wide recovery:')
    print(show.round(3).to_string(index=False))
    print('\nTK vs non-TK AUROC:')
    print(tab[tab.split != 'all'].pivot_table(index=['method', 'selection', 'ranking'],
                                              columns='split', values='auroc').round(3).to_string())
    print(f'\n  wrote {OUT / "pathway_localora_eval.parquet"}')


if __name__ == '__main__':
    main()
