"""kd_10 · The prediction deliverable: one workbook of the confidently-predicted kinase domains.

The workbook covers **every active kinase domain** (all species), tagged with whether each method has it
measured, predicted it, or neither. One workbook, `out/kd_predictions.xlsx`:

  description   a README (first sheet) documenting every easily-confused column and each sheet.
  info   one row per **active** kinase domain (every species) — its identity (kd_ID, gene, protein,
         organism, human?), a **category** (see below), **predicted_methods** (which methods predicted it)
         and **measured_methods** (which already have it measured — a domain is predicted only for the
         methods it *lacks*, so e.g. ERBB2, measured by PSPA + MLP-attr, appears only as a CDDM prediction),
         the novelty flags, then a **canonical** retrieval (group / nearest_kinases / nn_dist against the
         union of all characterised kinases; a measured kinase is its own nearest, nn_dist 0), then the
         **per-method** retrieval that produced each prediction ({method}_group / _neighbors / _nn_dist /
         _proximity_tier vs that method's own labelled set — blank both where the method already has the
         motif measured and where the domain was not predictable). proximity_tier grades each prediction
         high/intermediate/low by equal-distance thirds of the cutoff (kd_06b), on that method's nn_dist (raw
         proximity, not accuracy-calibrated).
  nonhuman_ortholog_near / nonhuman_diverged / human_measured_predicted / human_predicted   the PREDICTED
         info rows split by novelty category (same columns as info), each ordered nearest-first (ascending
         nn_dist) — so `human_predicted` is the genuinely novel dark-kinome list.
  not_predicted   the rest — active domains not predicted by any method: category "measured" (characterised
         but not predicted) or "unknown" (too far to predict); ordered nearest-first.
  PSPA / CDDM / MLP_attr   the predicted PSSM per domain (one sheet each), for the domains that method
         predicted.

**Category (not all predictions are equally novel).** For a PREDICTED domain the category is its novelty
class; a domain predicted by no method is "measured" (characterised by some method) or "unknown" (neither
measured nor predictable). Most predictable domains are trivial ortholog retrieval that should not inflate
the headline. **Human** kinases are split purely by gene name (human gene symbols are reliable): gene
matches a characterised kinase -> human_measured_predicted, else -> human_predicted. **Non-human** gene
symbols are unreliable (viral v-Abl, renamed species duplicates), so a non-human is trivial if its gene
matches OR it is near-identical to a characterised kinase (`nn_dist <= NEAR`):
  nonhuman_ortholog_near    non-human, gene-matched OR near-identical - trivial ortholog retrieval (bulk)
  nonhuman_diverged         non-human, gene NOT matched AND nn_dist > NEAR - genuinely distant, real
                            extrapolation. NOT called 'paralog': a gene symbol cannot tell a paralog from a
                            diverged ortholog, and the boundary is a soft `nn_dist` cut, so the per-row
                            `nn_dist` is the honest signal.
  human_measured_predicted  human, gene matches a characterised kinase - the SAME kinase, measured by some
                            method and predicted for the rest (a gap-fill, not novel; not a paralog)
  human_predicted           human, gene NOT matched - a paralog of its nearest characterised kinase,
                            predicted only; the genuinely novel dark-kinome list
The `human_predicted` rows are plausible (retrieved from the nearest characterised kinase), not validated —
they are dark precisely because no measured motif exists.

Three columns keep it honest (gene symbols are imperfect and the distance cut is soft):
  gene_match      does the domain's gene match a characterised kinase? (the human split, and one arm of
                  the non-human split) — so within nonhuman_ortholog_near you can see true gene-orthologs
                  vs the near-identical-only rows (v-Abl etc.).
  putative        likely non-functional — the UniProt protein name says "putative"/"pseudogene". A
                  human_predicted prediction on a pseudogene (e.g. PRKY) is not a real novel kinase.
                  Written ONLY in the human_predicted sheet (only those flags were reviewed).
  near_identical  nn_dist <= NEAR — the other arm of the non-human trivial rule; surfaced so the cut is
                  transparent.
So the genuinely novel set is category in {human_predicted, nonhuman_diverged} AND NOT putative.

Inputs   out/kd_pred_new_{pspa,cddm,mlp_attr}.parquet (kd_07 + kd_06b), out/kd_train_{...}_onehot.parquet
         (kd_03), out/kd_feat_onehot.parquet (kd_02a), kdata kd_uniprot
Outputs  out/kd_predictions.xlsx (sheets: description, info, {nonhuman_ortholog_near, nonhuman_diverged,
         human_measured_predicted, human_predicted}, not_predicted, PSPA, CDDM, MLP_attr),
         out/kd_prediction_categories.parquet (breakdown)

Run:  python nbs/kd_10_prediction_excel.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kd_util
import numpy as np
import pandas as pd
from paths import OUT

import kdata

METHODS = [('pspa', 'PSPA'), ('cddm', 'CDDM'), ('mlp_attr', 'MLP_attr')]
K = 5                              # nearest labelled kinases to list (kd_07's retrieval k)
NEAR = 3.0                         # nn_dist <= NEAR -> near-identical (trivial retrieval, whatever the gene);
                                   # ~<=4-5 differing aligned residues, e.g. viral v-Abl/v-Raf/v-Erbb
PUTATIVE = r'putative|pseudogene'  # protein-name flag for likely-non-functional (pseudogene / uncertain)
CAT_DESC = {                       # plain-language description of each category (used in the breakdown)
    'nonhuman_ortholog_near': f'non-human; the gene matches a characterized kinase, or the domain is '
                              f'near-identical to one (nn_dist <= {NEAR:g}), so retrieval is an ortholog '
                              f'lookup',
    'nonhuman_diverged': f'non-human; the gene matches no characterized kinase and nn_dist > {NEAR:g}, so '
                         f'the domain is genuinely distant from any characterized kinase',
    'human_measured_predicted': 'human; the gene matches a characterized kinase, so this is the same kinase '
                                'already measured by some method and predicted only for the method or '
                                'methods it lacks (for example ERBB2, measured by PSPA and MLP-attr, '
                                'predicted for CDDM)',
    'human_predicted': 'human; the gene matches no characterized kinase, so this is a paralog of its '
                       'nearest characterized kinase, predicted and not measured'}


def gene_of(kd_id):
    "Kinase gene name embedded in a kd_ID ('Q2M2I8_AAK1_HUMAN_KD1' -> 'AAK1')."
    parts = str(kd_id).split('_')
    return parts[1] if len(parts) > 1 else str(kd_id)


def classify(is_human, gene_match, near_identical):
    """Novelty category. **HUMAN** kinases are split purely by gene name — human gene symbols are reliable:
    gene matches a characterised kinase -> human_measured_predicted, else -> human_predicted. **NON-HUMAN** gene symbols are
    unreliable (viral v-Abl, renamed species duplicates), so a non-human is 'trivial' if its gene matches
    OR it is near-identical to a characterised kinase (nn_dist <= NEAR):
      nonhuman_ortholog_near    non-human, gene-matched OR near-identical - trivial ortholog retrieval
      nonhuman_diverged         non-human, gene NOT matched AND nn_dist > NEAR - genuine extrapolation
      human_measured_predicted  human, gene matches a characterised kinase - the SAME kinase, measured by
                                some method and predicted for the ones it lacks (a gap-fill, not novel)
      human_predicted           human, gene NOT matched - a paralog of its nearest characterised kinase,
                                predicted only, genuinely novel (dark kinome; `putative` flags pseudogenes)
    (`gene_match` and `near_identical` are also written as columns, so which arm placed each row is visible.)
    """
    human = np.asarray(is_human, bool)
    gm = np.asarray(gene_match, bool)
    nh_trivial = gm | np.asarray(near_identical, bool)
    return np.select([human & gm, human & ~gm, ~human & nh_trivial],
                     ['human_measured_predicted', 'human_predicted', 'nonhuman_ortholog_near'],
                     default='nonhuman_diverged')


def description_sheet(n_union, breakdown, n_total, n_notpred):
    "A README (first sheet): the easily-confused columns, then each sheet with its category count + meaning."
    cols = [
        ('WORKBOOK', 'Supplementary Data S6 | Predicted kinase-domain specificity profiles. Annotations '
                     'and predicted PSPA, CDDM and MLP-attribution PSSMs for kinase domains across '
                     'species, predicted only for domains lying within the sequence-proximity range of the '
                     f'{n_union} characterized kinases.'),
        ('', ''),
        ('COLUMNS', ''),
        ('kd_ID', 'domain identifier, formatted {UniProtAcc}_{gene}_{ORGANISM}_KD{n}'),
        ('gene / protein / organism', 'UniProt identity of the domain'),
        ('is_human', 'TRUE if the organism is Homo sapiens'),
        ('category', 'prediction status and novelty in one column. For predicted domains it is the novelty '
                     'class, defined in the category sheets below: nonhuman_ortholog_near, '
                     'nonhuman_diverged, human_measured_predicted or human_predicted. For the rest, '
                     '"measured" = characterized by some method but not predicted, and "unknown" = neither '
                     'measured nor within the proximity range required for a prediction.'),
        ('predicted_methods', 'the methods that predicted this domain, meaning the domain fell within the '
                              "proximity range and lacked that method's measurement. Blank = predicted by "
                              'no method.'),
        ('measured_methods', 'the methods that already have a measured motif for this domain. A domain is '
                             'predicted only for the methods it lacks, so ERBB2 = "PSPA, MLP_attr" means '
                             'only its CDDM motif is predicted. Blank = no measured motif by any method.'),
        ('gene_match', 'whether the domain gene matches a characterized kinase gene. This separates the '
                       'human categories (human_measured_predicted vs human_predicted) and is one arm of '
                       'the non-human ortholog rule.'),
        ('putative', 'TRUE if the UniProt protein name contains "putative" or "pseudogene", marking a '
                     'likely non-functional entry (for example PDPK2P, CSNK2A3, PRKY); such a '
                     'human_predicted row is not a novel kinase. Shown in the human_predicted sheet only, '
                     'as only those entries were reviewed.'),
        ('near_identical', f'TRUE if nn_dist <= {NEAR:g}, meaning the domain is near-identical to a '
                           'characterized kinase and its retrieval is an ortholog lookup regardless of the '
                           'gene name (this captures cases such as viral v-Abl and renamed species '
                           'duplicates). The other arm of the non-human ortholog rule.'),
        ('group / nearest_kinases / nn_dist', f'retrieval against the union of all {n_union} characterized '
                             'kinases across the three methods, rather than against any single method: the '
                             'k = 5 nearest domains on the one-hot alignment features. nearest_kinases = '
                             "the 5 nearest kinase genes, closest first; group = the nearest one's kinase "
                             'group; nn_dist = distance to the single nearest. A domain that is itself '
                             'measured by some method is its own nearest, so its nn_dist is 0 and it heads '
                             'nearest_kinases.'),
        ('PSPA_* / CDDM_* / MLP_attr_* ({group,neighbors,nn_dist,proximity_tier})',
                             "the per-method retrieval that produced that method's prediction: the same as "
                             "the columns above but against that method's own labeled set. "
                             '{method}_nn_dist is the distance that determines whether the domain is close '
                             'enough to predict, and {method}_proximity_tier grades it as high '
                             '(nn_dist <= 3.9), intermediate (<= 7.9) or low (<= 11.83), the '
                             "equal-distance thirds of the cutoff scored on that method's own nn_dist. "
                             'The tier is raw domain proximity and is not calibrated to accuracy, so a '
                             'domain can fall in different tiers for different methods. Blank when the '
                             'method already has the motif measured, or when the domain was outside the '
                             'proximity range for that method.'),
        ('', ''),
        ('SHEETS', ''),
        ('info', f'one row per active kinase domain ({n_total:,} total, all species), with all columns '
                 'above'),
    ]
    for _, b in breakdown.iterrows():                          # each predicted-category sheet: count + def
        cols.append((b.category, f'predicted-category sheet, {int(b.n):,} domains (median nn_dist '
                                 f'{b.median_nn_dist:.2f}). {b.description}'))
    cols.append(('not_predicted', f'the remaining {n_notpred:,} domains, not predicted by any method: '
                                  'category "measured" (characterized but not predicted) or "unknown" '
                                  '(outside the proximity range); ordered nearest first.'))
    cols.append(('PSPA / CDDM / MLP_attr',
                 'the predicted PSSM (residue x position) per domain, for that method'))
    return pd.DataFrame(cols, columns=['column / sheet', 'description'])


def retrieve(feat_all, feat_col, ref_ids, query_ids, exclude_self=False):
    """k nearest labelled kinases per query domain: nearest group, gene names (ranked), nn_dist.

    exclude_self=True drops a query that is itself in the reference set — a measured kinase is its own
    nearest (distance 0), so this makes nn_dist mean 'distance to the nearest OTHER characterised kinase'.
    """
    from sklearn.neighbors import NearestNeighbors
    ref_ids = list(ref_ids)
    ref_arr = np.array(ref_ids)
    ref_gene = np.array([gene_of(k) for k in ref_ids])
    ref_group = kd_util.kinase_taxonomy(pd.Index(ref_ids)).set_index('kinase')['group'].reindex(ref_ids).to_numpy()
    k = min(K + (1 if exclude_self else 0), len(ref_ids))
    nn = NearestNeighbors(n_neighbors=k).fit(feat_all.loc[ref_ids, feat_col].to_numpy(float))
    query_ids = list(query_ids)
    dist, idx = nn.kneighbors(feat_all.loc[query_ids, feat_col].to_numpy(float))
    grp, neigh, nnd = [], [], []
    for qi, q in enumerate(query_ids):
        ii, dd = idx[qi], dist[qi]
        if exclude_self:
            keep = ref_arr[ii] != q
            ii, dd = ii[keep][:K], dd[keep][:K]
        else:
            ii, dd = ii[:K], dd[:K]
        grp.append(ref_group[ii[0]])
        neigh.append(', '.join(ref_gene[j] for j in ii))
        nnd.append(round(float(dd[0]), 2))
    return pd.DataFrame({'group': grp, 'neighbors': neigh, 'nn_dist': nnd},
                        index=pd.Index(query_ids, name='kd_ID'))


def main():
    feat_all = pd.read_parquet(OUT / 'kd_feat_onehot.parquet')
    feat_col = list(feat_all.columns)

    predictable, pssm_sheets, refs, conf, union_ref = {}, {}, {}, {}, set()
    for target, name in METHODS:
        df, _, target_col = kd_util.load_train(target, 'onehot')
        refs[name] = df.iloc[:, 0].tolist()
        union_ref |= set(refs[name])
        pred = pd.read_parquet(OUT / f'kd_pred_new_{target}.parquet')
        q = list(pred.index[pred.predictable.to_numpy()])
        predictable[name] = q
        pssm_sheets[name] = pred.loc[q, target_col]
        conf[name] = pred.loc[q, ['proximity_tier']]   # per-prediction proximity tier (kd_06b)
        print(f'  {name}: {len(q)} predictable domains', flush=True)

    domains = sorted(feat_all.index)                           # ALL active kinase domains (the universe)
    pred_union = set().union(*predictable.values())            # predictable by >=1 method
    measured = {name: set(refs[name]) for _, name in METHODS}

    u = kdata.load('kd_uniprot').set_index('kd_ID')
    meta_cols = {'Gene Names (primary)': 'gene', 'Protein names': 'protein', 'Organism': 'organism'}
    info = u.reindex(domains)[list(meta_cols)].rename(columns=meta_cols)
    info['is_human'] = info.organism.str.contains('Homo sapiens', na=False)

    canon = retrieve(feat_all, feat_col, sorted(union_ref), domains)   # vs all characterised (self included)
    info = info.join(canon.rename(columns={'neighbors': 'nearest_kinases'}))

    # status: which methods already MEASURED this domain, and which PREDICTED it (predictable & lacking it)
    measured_methods = [', '.join(n for _, n in METHODS if d in measured[n]) for d in domains]
    predicted_methods = [', '.join(n for _, n in METHODS if d in set(predictable[n])) for d in domains]

    # category — novelty for the PREDICTED domains (trivial = characterised gene OR near-identical; the
    # near-identical arm reclaims orthologs the gene symbol misses, viral v-Abl etc.), else a status:
    #   measured = characterised by some method but not predicted;  unknown = neither (too far to predict).
    characterized = {gene_of(k).upper() for k in union_ref}
    gene_match = np.array([gene_of(d).upper() in characterized for d in domains])
    near = (info.nn_dist <= NEAR).to_numpy()
    novelty = classify(info.is_human.to_numpy(), gene_match, near)
    is_pred = np.array([d in pred_union for d in domains])
    has_meas = np.array([bool(m) for m in measured_methods])
    category = np.where(is_pred, novelty, np.where(has_meas, 'measured', 'unknown'))

    at = list(info.columns).index('is_human') + 1
    info.insert(at, 'category', category)
    info.insert(at + 1, 'predicted_methods', predicted_methods)   # methods that predicted it (blank if none)
    info.insert(at + 2, 'measured_methods', measured_methods)     # methods with a measured motif (blank if none)
    info.insert(at + 3, 'gene_match', gene_match)                 # gene matches a characterised kinase?
    info.insert(at + 4, 'putative', info.protein.str.contains(PUTATIVE, case=False, na=False))
    info.insert(at + 5, 'near_identical', near)

    for target, name in METHODS:                               # per-method retrieval + confidence (that produced it)
        m = retrieve(feat_all, feat_col, refs[name], predictable[name]).join(conf[name])
        info = info.join(m.rename(columns={c: f'{name}_{c}' for c in m.columns}))

    NOVELTY = ['nonhuman_ortholog_near', 'nonhuman_diverged', 'human_measured_predicted', 'human_predicted']
    breakdown = (info[info.category.isin(NOVELTY)].groupby('category')
                 .agg(n=('category', 'size'), median_nn_dist=('nn_dist', 'median'))
                 .reindex(NOVELTY).dropna(how='all').reset_index())
    breakdown['n'] = breakdown['n'].astype(int)
    breakdown.insert(1, 'description', breakdown.category.map(CAT_DESC))
    breakdown.to_parquet(OUT / 'kd_prediction_categories.parquet')
    novel_human = info[info.category == 'human_predicted'].sort_values('nn_dist')

    def as_sheet(df, xl, sheet, keep_putative=False):           # kd_ID from the index, back to a column
        "putative is shown ONLY in the human_predicted sheet (only those flags were manually reviewed)."
        if not keep_putative and 'putative' in df.columns:
            df = df.drop(columns='putative')
        df.reset_index().rename(columns={'index': 'kd_ID'}).to_excel(xl, sheet_name=sheet, index=False)

    n_notpred = int((~info.category.isin(NOVELTY)).sum())      # measured-not-predicted + unknown (the rest)
    path = OUT / 'kd_predictions.xlsx'
    with pd.ExcelWriter(path, engine='openpyxl') as xl:
        description_sheet(len(union_ref), breakdown, len(info), n_notpred).to_excel(
            xl, sheet_name='description', index=False)
        ws = xl.sheets['description']                          # readable widths + wrapped text
        ws.column_dimensions['A'].width, ws.column_dimensions['B'].width = 42, 130
        for cell in ws['B']:
            cell.alignment = cell.alignment.copy(wrap_text=True)
        as_sheet(info, xl, 'info')
        for cat in NOVELTY:                                    # one sheet per predicted category, nearest-first
            sub = info[info.category == cat].sort_values('nn_dist')
            if len(sub):
                as_sheet(sub, xl, cat, keep_putative=(cat == 'human_predicted'))   # putative only there
        as_sheet(info[~info.category.isin(NOVELTY)].sort_values('nn_dist'), xl, 'not_predicted')  # the rest
        for _, name in METHODS:
            pssm_sheets[name].reset_index().to_excel(xl, sheet_name=name, index=False)
    print(f'\nwrote {path}  ({len(info):,} active domains; {len(pred_union):,} predicted, {n_notpred:,} not; '
          f'sheets: description, info, ' + ', '.join(NOVELTY) + ', not_predicted, '
          + ', '.join(n for _, n in METHODS) + ')')
    print('\ncategory breakdown:\n' + breakdown.to_string(index=False))

    func = novel_human[~novel_human.putative]
    print(f'\nhuman dark kinome: {len(novel_human)} predicted, '
          f'{int(novel_human.putative.sum())} putative/pseudogene flagged -> '
          f'{len(func)} non-pseudogene human predictions (functionality/accuracy unvalidated)')
    print('  non-pseudogene :', ', '.join(func.gene))
    print('  putative       :', ', '.join(novel_human[novel_human.putative].gene))
    genuine = info[(info.category == 'nonhuman_diverged') & ~info.putative]
    print(f'genuine non-human extrapolation (nonhuman_diverged, not putative): {len(genuine)}')


if __name__ == '__main__':
    main()
