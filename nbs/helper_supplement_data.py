"""helper · Build the paper's Supplementary Data workbooks.

Assembles seven figure-aligned Excel workbooks into `supplement_table/`, each opening with a
README sheet that documents every other sheet, its source script, sample size and the meaning of
easily-confused columns. Every number is read from the persisted `out/` / `pssm/` / kdata store or
recomputed with the same helpers the paper figures use; nothing is re-fit here.

  S1  kinase_annotation           the 523-kinase annotation table + per-kinase method coverage
  S2  kinase_substrate_dataset    raw per-pair records, per-kinase counts, unique-site table with num_kin,
                                  cutoff coverage
  S3  kinase_prediction_benchmark Fig. 2: out-of-fold CV scores overall + by group + paired stats
  S4  pssm_specificity            Fig. 3/5/6: CDDM / MLP-attr / PSPA / surface-display PSSMs, method
                                  agreement with PSPA, specificity index, flanking-pY priming
  S5  kinase_domain_model         Fig. 7 / Supp. Fig. 2: model x feature CV, leave-one-out per kinase,
                                  proximity cutoff, external non-human validation, taxonomy silhouette
  S6  predicted_kinase_domains    the kd_10 deliverable workbook (copied verbatim)
  S7  pathway_recovery            Fig. 8: kinome-wide recovery per method, per-kinase CDDM, examples

No held-out test set is involved anywhere: the benchmark (S3) and the kinase-domain grid (S5) are both
scored by repeated leakage-safe cross-validation, out of fold. Titles and READMEs say so.

Excel caps a cell at 32,767 characters, so full-length protein sequences are never written to a sheet
(the raw S2 per-pair records keep the +/-20 site window and drop the full-length sequence columns).

Inputs   kdata (kinase_info, ks_unique, cddm, pspa, pspa_enrich), out/*.parquet|csv|xlsx, pssm/sd_*.parquet
Outputs  supplement_table/Supplementary_Data_S{1..7}_*.xlsx

Run:  python nbs/helper_supplement_data.py            # all seven
      python nbs/helper_supplement_data.py S1 S4      # only some
"""
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from openpyxl.styles import Alignment
from paths import OUT, PSSM

import kdata

SUPP = Path(__file__).resolve().parent / 'supplement_table'
EXCEL_CELL_LIMIT = 32_767
POS_COL = re.compile(r'^(-?\d+)([A-Za-z])$')     # a PSSM column name: "{position}{residue}", e.g. -3P, 0y, 4t


# ----------------------------------------------------------------------------------------------------
# writer helpers
# ----------------------------------------------------------------------------------------------------
def _check_cells(sheet, df):
    "Refuse to write a sheet whose longest cell would be truncated by Excel."
    for c in df.columns:
        try:
            m = int(df[c].astype(str).str.len().max())
        except Exception:
            continue
        if m > EXCEL_CELL_LIMIT:
            sys.exit(f'{sheet}.{c}: cells up to {m:,} chars exceed Excel {EXCEL_CELL_LIMIT:,} - trim or write CSV')


def write_book(fname, sheets, readme):
    """Write an ordered {sheet_name: df} workbook, README first.

    `readme` is a list of (sheet, description) rows; it is rendered as the first sheet with wrapped,
    readable column widths. Every sheet name is truncated to Excel's 31-char limit.
    """
    SUPP.mkdir(exist_ok=True)
    rdf = pd.DataFrame(readme, columns=['sheet / column', 'description'])
    out = SUPP / fname
    with pd.ExcelWriter(out, engine='openpyxl') as xl:
        rdf.to_excel(xl, sheet_name='README', index=False)
        ws = xl.sheets['README']
        ws.column_dimensions['A'].width, ws.column_dimensions['B'].width = 40, 135
        for cell in ws['B']:
            cell.alignment = Alignment(wrap_text=True, vertical='top')
            if ws.cell(cell.row, 1).value == 'agreement_vs_PSPA':
                ws.row_dimensions[cell.row].height = 130
        for name, df in sheets.items():
            sheet = name[:31]
            _check_cells(sheet, df)
            df.to_excel(xl, sheet_name=sheet, index=False)
            if 'mean_pos_kld' in df.columns:
                ws = xl.sheets[sheet]
                col = int(df.columns.get_loc('mean_pos_kld')) + 1
                ws.column_dimensions[ws.cell(1, col).column_letter].width = 21
                for row in ws.iter_rows(min_row=2, min_col=col, max_col=col):
                    row[0].number_format = '0.000000'
    n = sum(len(d) for d in sheets.values())
    print(f'  wrote {out.name}  ({len(sheets)} sheets + README, {n:,} data rows, '
          f'{out.stat().st_size / 1e6:.1f} MB)')


def flank(df, w=5, keep_pos0=True):
    "Slice a (kinase x position-residue) PSSM to the +/-w flank, optionally keeping position 0."
    cols = []
    for c in df.columns:
        m = POS_COL.match(str(c))
        if not m:
            continue
        p = int(m.group(1))
        if -w <= p <= w and (keep_pos0 or p != 0):
            cols.append(c)
    return df[cols]


# ----------------------------------------------------------------------------------------------------
# S1 - kinase annotation
# ----------------------------------------------------------------------------------------------------
def build_s1():
    info = kdata.load('kinase_info')
    keep = ['kinase', 'uniprot', 'gene', 'group', 'group_old', 'family', 'subfamily',
            'in_pspa', 'in_pspa_st', 'in_pspa_tyr', 'in_cddm', 'pseudo', 'active_D1_D2', 'kd_ID',
            'pspa_category_big', 'pspa_category_small']
    ann = info[[c for c in keep if c in info.columns]].copy()

    readme = [
        ('WORKBOOK', 'Supplementary Data S1 | Kinase annotations. Reference annotations for every kinase '
                     'used in this study. The "kinase" / "gene" identifier keys every other supplementary '
                     'workbook.'),
        ('kinase_info', f'{len(ann)} kinases on the kinome tree. group = the kinase group of Modi and '
                        'Dunbrack, used throughout (TK = tyrosine kinase); group_old = the traditional '
                        'Manning label. in_pspa / in_cddm flag whether a kinase has a measured PSPA / CDDM '
                        'motif; in_pspa_tyr marks a tyrosine-array (dual-specificity) measurement. pseudo / '
                        'active_D1_D2 flag pseudokinases and catalytically active domains; pspa_category '
                        'is the PSPA specificity class. Per-kinase substrate-site counts and source '
                        'contributions are in Supplementary Data S2. Full-length protein sequences are '
                        'omitted.'),
    ]
    write_book('Supplementary_Data_S1_kinase_annotation.xlsx', {'kinase_info': ann}, readme)


# ----------------------------------------------------------------------------------------------------
# S2 - kinase-substrate dataset
# ----------------------------------------------------------------------------------------------------
def build_s2():
    ks = kdata.ks_dataset(thr=None)

    # per-kinase counts, site-level: one row per distinct substrate site (substrate + position), with how
    # many of a kinase's sites are reported by PSP / Sugiyama. Keyed by the canonical kinase name.
    ks = ks.assign(is_psp=ks.source.str.contains('PSP', regex=False),
                   is_sug=ks.source.str.contains('Sugiyama', regex=False))
    sl = ks.groupby(['kinase_protein', 'sub_site']).agg(
        psp=('is_psp', 'any'), sug=('is_sug', 'any'), group=('kinase_group', 'first')).reset_index()
    per_kin = sl.groupby('kinase_protein').agg(
        group=('group', 'first'),
        n_uniqueID_sites=('sub_site', 'size'),
        n_from_PSP=('psp', 'sum'),
        n_from_Sugiyama=('sug', 'sum')).reset_index().rename(columns={'kinase_protein': 'kinase'})
    per_kin['n_from_PSP'] = per_kin.n_from_PSP.astype(int)
    per_kin['n_from_Sugiyama'] = per_kin.n_from_Sugiyama.astype(int)
    per_kin['pct_from_PSP'] = (100 * per_kin.n_from_PSP / per_kin.n_uniqueID_sites).round(1)
    per_kin['pct_from_Sugiyama'] = (100 * per_kin.n_from_Sugiyama / per_kin.n_uniqueID_sites).round(1)
    # n_seqdedup_sites: sites actually used to build the CDDM motif (num_kin <= 40, case-insensitively
    # deduplicated on the sequence window), the same basis motif_01 uses
    d40 = ks[ks.num_kin <= 40].copy()
    d40['u'] = d40.site_seq.str.upper()
    n_dedup = d40.drop_duplicates(['kinase_protein', 'u']).kinase_protein.value_counts()
    per_kin['n_seqdedup_sites'] = per_kin.kinase.map(n_dedup).astype('Int64')
    per_kin = per_kin[['kinase', 'group', 'n_uniqueID_sites', 'n_from_PSP', 'n_from_Sugiyama',
                       'pct_from_PSP', 'pct_from_Sugiyama', 'n_seqdedup_sites']].sort_values(
        'n_uniqueID_sites', ascending=False)

    # unique-site table with num_kin, collapsing the per-kinase one-hot into a compact kinase list
    ksu = kdata.load('ks_unique')
    meta = ['sub_site', 'sub_genes', 'acceptor', 'num_kin', 'bin', 'source_combine', 'site_seq']
    onehot = [c for c in ksu.columns if c not in meta and c not in ('site_seq',)
              and c not in ('sub_site', 'num_kin', 'bin', 'sub_genes', 'source_combine', 'acceptor')]
    genes = np.array([c.split('_', 1)[1] if '_' in c else c for c in onehot])
    arr = ksu[onehot].to_numpy(bool)
    kin_list = [', '.join(genes[row]) for row in arr]
    sites = ksu[[c for c in meta if c in ksu.columns]].copy()
    sites['kinases'] = kin_list
    sites = sites.sort_values('num_kin', ascending=False)

    # num_kin distribution (Supp. Fig. 1a) and per-cutoff kinase coverage (Supp. Fig. 1f)
    dist = (ksu.groupby('bin').agg(n_sites=('sub_site', 'size'))
            .reset_index().assign(pct=lambda d: (100 * d.n_sites / d.n_sites.sum()).round(2)))
    cutoffs = [5, 10, 20, 40, 60, 80, 100, 150, 200, 300]
    rows = []
    for thr in cutoffs:
        sub = ksu[ksu.num_kin <= thr]
        counts = np.asarray(sub[onehot].sum(axis=0))          # sites per kinase within the cutoff
        rows.append({'num_kin_cutoff': thr,
                     'n_sites': int(len(sub)),
                     'n_kinases_ge40_sites': int((counts >= 40).sum())})
    coverage = pd.DataFrame(rows)

    # phospho-priming prevalence: unique sites with >=1 flanking pS/pT/pY (lowercase s/t/y) in the window
    def _primed(seq, w):
        if not isinstance(seq, str) or len(seq) < 41:
            return False
        c = 20
        return any(ch in 'sty' for ch in seq[c - w:c] + seq[c + 1:c + w + 1])
    prows = []
    for acc in ['S', 'T', 'Y', 'all']:
        d = ksu if acc == 'all' else ksu[ksu.acceptor == acc]
        rec = {'acceptor': acc, 'n_uniqueID_sites': len(d)}     # unique sites by substrate + acceptor + position
        for w in (5, 7):
            n = int(d.site_seq.apply(lambda s: _primed(s, w)).sum())
            rec[f'n_with_flank_primed_pm{w}'] = n
            rec[f'pct_with_flank_primed_pm{w}'] = round(100 * n / len(d), 1)
        prows.append(rec)
    primed = pd.DataFrame(prows)

    # per-source contribution / overlap, from the |-joined source tag on each kinase-substrate pair
    SOURCES = ['Sugiyama', 'PSP', 'EPSD', 'iPTMNet', 'SIGNOR', 'ELM', 'GPS6']
    src = ks.source.fillna('')
    srows = [{'source': s,
              'pairs_containing': int(src.str.contains(s, regex=False).sum()),
              'exclusive_to_source': int((src == s).sum())} for s in SOURCES]
    source_contrib = pd.DataFrame(srows).sort_values('pairs_containing', ascending=False)

    # raw per-pair records as a sheet, minus the full-length sequence columns (they exceed Excel's cell cap)
    # and the internal is_psp/is_sug helper flags added above
    drop = [c for c in ('substrate_sequence', 'substrate_phosphoseq', 'human_uniprot_sequence',
                        'is_psp', 'is_sug') if c in ks.columns]
    raw = ks.drop(columns=drop)

    readme = [
        ('WORKBOOK', 'Supplementary Data S2 | Kinase-substrate dataset. The integrated kinase-substrate '
                     'collection (PSP, Sugiyama et al., ELM, SIGNOR, GPS6, KiNET) after UniProt remapping, '
                     'site validation and deduplication. Supports Fig. 1b and Supplementary Fig. 1.'),
        ('raw_pairs', f'{len(raw):,} kinase-substrate-site records, one per row. Full-length protein '
                      'sequences are omitted; the +/-20 AA window around each site is kept in site_seq. '
                      'num_kin = number of distinct kinases annotated to the substrate site.'),
        ('per_kinase_counts', f'{len(per_kin)} kinases, keyed by canonical name. Two site counts on '
                              'different deduplication bases: n_uniqueID_sites = unique sites by substrate '
                              'UniProt + acceptor + position; n_seqdedup_sites = unique sites by sequence '
                              'window (case-insensitive), restricted to num_kin <= 40 (the basis used to '
                              'build the CDDM motif; blank if none). The two differ because different '
                              'positions can share one sequence window and because CDDM drops highly '
                              'promiscuous (num_kin > 40) sites. n_from_PSP / n_from_Sugiyama = how many of '
                              'the n_uniqueID_sites are reported by PSP / Sugiyama et al.; pct_from_PSP / '
                              'pct_from_Sugiyama are the same as percentages.'),
        ('unique_sites', f'{len(sites):,} unique substrate sites (deduplicated on substrate + acceptor + '
                         'position). sub_site = substrate + acceptor + position key; site_seq = the +/-20 '
                         'sequence window (lowercase s/t/y = flanking phospho-residues); num_kin = number '
                         'of distinct kinases; bin = promiscuity class; source_combine = a three-way source '
                         'label (Sugiyama / Non-Sugiyama / Both, as used in Fig. 2c); kinases = the '
                         'comma-separated list of kinases annotated to the site.'),
        ('num_kin_distribution', 'Site count and percentage per promiscuity bin (1 / 2-10 / 11-100 / '
                                 '101-300), the data behind Supplementary Fig. 1a.'),
        ('cutoff_coverage', 'For each candidate num_kin training cutoff: sites retained and how many kinases '
                            'still have at least 40 substrate sites (Supplementary Fig. 1f). The cutoff used '
                            'throughout is num_kin <= 40.'),
        ('source_contribution', 'Per source, the number of kinase-substrate-site pairs it contributes and '
                                'how many are exclusive to it (reported by no other source). Documents '
                                'dataset composition and cross-source overlap.'),
        ('phospho_primed', 'Phospho-priming prevalence per acceptor (S/T/Y, and all). n_uniqueID_sites = '
                           'unique substrate sites counted by substrate UniProt + acceptor + position (the '
                           'sub_site key, not sequence-deduplicated). n_with_flank_primed_pm5 / _pm7 = how '
                           'many of those sites carry at least one flanking pS/pT/pY within +/-5 / +/-7 of '
                           'the acceptor (position 0 excluded); pct_with_flank_primed_pm5 / _pm7 are the '
                           'same as percentages. Flanking phosphorylation is annotated by cross-referencing '
                           'every known phosphosite on the substrate. Related to Fig. 6a, b, which count '
                           'sequence-window-deduplicated sites, so those figure n values differ from '
                           'n_uniqueID_sites here.'),
    ]
    write_book('Supplementary_Data_S2_kinase_substrate_dataset.xlsx',
               {'raw_pairs': raw, 'per_kinase_counts': per_kin, 'unique_sites': sites,
                'num_kin_distribution': dist, 'cutoff_coverage': coverage,
                'source_contribution': source_contrib, 'phospho_primed': primed}, readme)


# ----------------------------------------------------------------------------------------------------
# S3 - kinase-assignment benchmark (Fig. 2)
# ----------------------------------------------------------------------------------------------------
def build_s3():
    import scoring_04d_figures as f4
    import scoring_04e_stats_table as f4e
    import scoring_fig2_panels as f2
    import scoring_util as su

    pairs = f2.load_pairs_full(su.NK_MAIN)
    overall = f4.overall_stats(pairs).round(4).rename(columns={
        'micro': 'micro_recall@10', 'micro_lo': 'micro_recall@10_lo', 'micro_hi': 'micro_recall@10_hi',
        'macro': 'macro_recall@10', 'macro_lo': 'macro_recall@10_lo', 'macro_hi': 'macro_recall@10_hi'})
    g_micro = f4.group_stats(pairs, macro=False).round(4).rename(columns={'mean': 'micro_recall@10'})
    g_macro = f4.group_stats(pairs, macro=True).round(4).rename(columns={'mean': 'macro_recall@10'})
    paired = f4e.paired_table(pairs)
    for c in ['median_delta_recall@10', 'delta_lo', 'delta_hi']:
        paired[c] = paired[c].round(4)
    counts = f4e.held_out_counts().rename(columns={'n_sites': 'n_uniqueID_sites',
                                                   'n_windows': 'n_unique_seq_windows'})

    readme = [
        ('WORKBOOK', 'Supplementary Data S3 | Kinase-assignment benchmark (Fig. 2). Rank the shared pool of '
                     '295 candidate kinases (219 S/T, 76 Tyr) for each substrate site. There is no held-out '
                     'test set: scores are out of fold over three repeated leakage-safe 5-fold '
                     'cross-validation partitions, with sites sharing a sequence window kept in one fold. '
                     'Reported on the reliably annotated subset of sites (at most 10 annotated kinases per '
                     'site). Uncertainty = kinase cluster-bootstrap 95% CI, taking the kinase as the '
                     'biological replicate.'),
        ('overall', 'Per branch (S/T, Tyr) x method: AUCDF, micro_recall@10 (pooled over sites) and '
                    'macro_recall@10 (averaged over kinases), each with its 95% CI in the matching _lo/_hi '
                    'columns. Nine methods (Random, Dummy, PSPA, PSPA pct, CDDM, CDDM pct, CDDM-seq '
                    'Linear/MLP/MLP+PSPA).'),
        ('by_group_micro', 'Micro recall@10 (pooled over sites) per kinase group x method, mean + 95% CI '
                           '(lo/hi). Behind Fig. 2e.'),
        ('by_group_macro', 'Macro recall@10 (averaged over kinases) per kinase group x method, mean + 95% CI '
                           '(lo/hi). Behind Fig. 2f.'),
        ('paired_comparisons', 'The CDDM-seq MLP + PSPA model vs each prespecified PSSM baseline: median '
                               'per-kinase delta recall@10 + 95% CI, paired Wilcoxon p (p_raw), '
                               'Holm-adjusted within branch (p_holm).'),
        ('held_out_counts', 'Evaluated counts per branch and subset: n_kinases; n_uniqueID_sites (substrate '
                            'sites by substrate + acceptor + position, the recall metric unit); '
                            'n_unique_seq_windows (distinct uppercase sequence windows among them).'),
    ]
    write_book('Supplementary_Data_S3_kinase_prediction_benchmark.xlsx',
               {'overall': overall, 'by_group_micro': g_micro, 'by_group_macro': g_macro,
                'paired_comparisons': paired, 'held_out_counts': counts}, readme)


# ----------------------------------------------------------------------------------------------------
# S4 - PSSMs, cross-method agreement, specificity, pY priming (Fig. 3/5/6)
# ----------------------------------------------------------------------------------------------------
def build_s4():
    cddm = flank(kdata.load('cddm')).reset_index(names='kinase')
    mlp = flank(pd.read_parquet(OUT / 'mlp_attr_pssm_full.parquet')).reset_index(names='kinase')
    pspa_norm = flank(kdata.load('pspa')).reset_index(names='kinase')            # as published
    pspa_scale = flank(kdata.load('pspa_scale')).reset_index(names='kinase')      # each position sums to 1
    pspa_enrich = flank(kdata.load('pspa_enrich')).reset_index(names='kinase')

    # surface display: stack the five matrices, each sliced to the +/-5 flank, labelled by the published
    # library design and matrix type rather than by the internal file name
    sd_files = {'sd_freq_p1': ('pTyr', 'frequency'), 'sd_ptyr_p1': ('pTyr', 'enrichment'),
                'sd_freq_p2': ('pTyr-Var', 'frequency'), 'sd_ptyrvar_p2': ('pTyr-Var', 'enrichment'),
                'sd_x5yx5_p2': ('X5-Y-X5', 'enrichment')}
    sd_parts = []
    for f, (library, kind) in sd_files.items():
        m = flank(pd.read_parquet(PSSM / f'{f}.parquet')).reset_index(names='kinase')
        m.insert(0, 'library', library)
        m.insert(1, 'type', kind)
        sd_parts.append(m)
    sd = pd.concat(sd_parts, ignore_index=True)

    # CDDM log-odds is not shown in the paper (Fig 5 uses 5 methods; Fig 3b drops log-odds), so exclude it
    LOGODDS = 'CDDM log-odds'
    vs_pspa = pd.concat([pd.read_csv(OUT / 'compare_vs_pspa_bygroup.csv'),
                         pd.read_csv(OUT / 'compare_vs_pspa.csv')], ignore_index=True)
    vs_pspa = vs_pspa[vs_pspa.method != LOGODDS].reset_index(drop=True).rename(
        columns={'spearman': 'mean_pos_spearman', 'top5': 'top5_overlap', 'ap': 'ap_at5'})
    # Response Fig. R1 uses the own-overlap alphabet with priming, not shared12's AA20.
    kld = pd.read_csv(OUT / 'motif_cddm_pspa_kld_scores.csv')
    kld = kld.loc[kld.alpha.eq(0.001), ['kinase', 'kld']].rename(
        columns={'kld': 'mean_pos_kld'})
    kld = kld.assign(method='CDDM freq', set='own_overlap')
    vs_pspa = vs_pspa.merge(kld, on=['method', 'kinase', 'set'], how='left',
                            validate='one_to_one')
    if vs_pspa.mean_pos_kld.notna().sum() != len(kld):
        raise ValueError('KLD cohort does not match the S4 CDDM own-overlap rows.')
    method_matrix = pd.read_csv(OUT / 'compare_methods_matrix.csv').rename(
        columns={'Unnamed: 0': 'metric', 'Unnamed: 1': 'method'})
    method_matrix = method_matrix[method_matrix.method != LOGODDS].drop(columns=LOGODDS)
    spec = pd.read_parquet(OUT / 'specificity_metrics.parquet').round(4)
    priming = pd.read_csv(OUT / 'priming_rationale.csv')

    readme = [
        ('WORKBOOK', 'Supplementary Data S4 | PSSMs and substrate specificity (Fig. 3, 5, 6). PSSMs are '
                     'given over the +/-5 flank, keeping position 0 (the phospho-acceptor). Rows are '
                     'kinases; columns are "{position}{residue}"; lowercase s, t and y denote pS, pT and '
                     'pY annotation tokens.'),
        ('CDDM_freq', f'{len(cddm)} kinases. Observed-substrate frequency PSSM (distinguishes pS/pT/pY).'),
        ('MLP_attr', f'{len(mlp)} kinases. CDDM MLP-attribution PSSM from in silico saturation mutagenesis; '
                     'signed (do not softmax), pS/pT distinct.'),
        ('PSPA_norm', f'{len(pspa_norm)} kinases. The normalized PSPA matrices as published in the S/T and '
                      'Tyr kinase specificity atlases, the starting point for the two representations '
                      'below. PSPA measures pT and duplicates it as pS at flanking positions.'),
        ('PSPA_scale', f'{len(pspa_scale)} kinases. PSPA with the values at each measured position '
                       'normalized to sum to 1.'),
        ('PSPA_enrich', f'{len(pspa_enrich)} kinases. PSPA signed log2 enrichment vs the per-position '
                        'median (the flank representation shown in Fig. 3c / 5).'),
        ('surface_display', 'Bacterial surface-display PSSMs for tyrosine kinases. Display peptides are not '
                            'pre-phosphorylated, so these matrices cover the 20 standard AAs only. '
                            '"library" names the published library design (pTyr, pTyr-Var, X5-Y-X5); '
                            '"type" is frequency or enrichment.'),
        ('agreement_vs_PSPA', 'Per-kinase agreement with PSPA over the flank (position 0 excluded). '
                              'mean_pos_spearman = mean per-position Spearman correlation; top5_overlap = '
                              'fraction of PSPA\'s 5 strongest flank cells recovered; ap_at5 = average '
                              'precision at 5 (AP@5). "set" = own_overlap (the per-group set behind '
                              'Fig. 3b, with pS/pT/pY kept) or shared12 (the 12 Tyr kinases of Fig. 5, '
                              'scored on the 20 standard AAs so surface display is comparable). '
                              'mean_pos_kld = mean per-position Kullback-Leibler divergence '
                              'KL(PSPA || CDDM) in nats, where lower values mean more similar '
                              'distributions; reported for the CDDM freq own_overlap rows only, other '
                              'rows are blank. At each shared flank position both profiles are normalized '
                              'to sum to one and smoothed as (1 - a) * p + a / N, with a = 0.001 and N the '
                              'number of shared residue features (pS, pT and pY kept separate), then '
                              'averaged equally over positions.'),
        ('method_matrix', 'Method x method agreement over the 12 shared Tyr kinases (Fig. 5a, d): mean '
                          'per-position Spearman and top-5 overlap between CDDM freq / MLP-attr / PSPA / '
                          'SD freq / SD enrich.'),
        ('spec_index', 'Per-kinase, per-method scale-free motif sharpness: gini, kurtosis, top3_share, '
                       'sharp_mean/max, eff_positions and the composite spec_index (Fig. 4a; the six '
                       "components are defined in Methods). Computed on each method's raw values so PSPA, "
                       'CDDM and MLP-attr sit on one footing.'),
        ('flanking_pY_priming', 'Fig. 6a-d, one row per flanking position. ks_ysite_p{S,T,Y}_freq = '
                                'fraction of observed unique Y-acceptor sites carrying pS / pT / pY at that '
                                'position. tk_pY_mean_scaled / tk_pST_mean_scaled = mean scaled PSPA '
                                'preference for pY / pS-or-pT across the 78 Tyr kinases at that position. '
                                'tk_n_pY_rank1 / tk_n_pST_rank1 = number of the 78 Tyr kinases whose top '
                                'residue at that position is pY / pS-or-pT; tk_n_pY_top5 / tk_n_pST_top5 = '
                                'number with pY / pS-or-pT among their 5 strongest. Overall 66 of 78 (85%) '
                                'rank pY first at one or more of -1/+1/+2.'),
    ]
    write_book('Supplementary_Data_S4_pssm_specificity.xlsx',
               {'CDDM_freq': cddm, 'MLP_attr': mlp, 'PSPA_norm': pspa_norm, 'PSPA_scale': pspa_scale,
                'PSPA_enrich': pspa_enrich,
                'surface_display': sd, 'agreement_vs_PSPA': vs_pspa, 'method_matrix': method_matrix,
                'spec_index': spec, 'flanking_pY_priming': priming}, readme)


# ----------------------------------------------------------------------------------------------------
# S5 - kinase-domain model (Fig. 7 / Supp. Fig. 2)
# ----------------------------------------------------------------------------------------------------
def _melt_model_feature():
    "Reshape kd_model_selection.xlsx's per-(target,subset) model x feature matrices into one tidy table."
    xl = pd.ExcelFile(OUT / 'kd_model_selection.xlsx')
    tidy = []
    for s in xl.sheet_names:
        if s == 'selected_cell':
            continue
        target, subset = s.rsplit('_', 1)
        raw = pd.read_excel(xl, s)
        metric = None
        for _, r in raw.iterrows():
            first = r['model']
            if isinstance(first, str) and pd.isna(r.get('onehot')):    # a metric header row
                metric = first.split(' (')[0]
                continue
            for feat in ['onehot', 'onehot_pca', 'esm', 't5']:
                tidy.append({'target': target, 'subset': subset, 'metric': metric,
                             'model': first, 'feature': feat, 'value': r[feat]})
    return pd.DataFrame(tidy)


def _clean_validation(path, id_name):
    "The kd_09 CSVs carry a two-row header (method / metric); flatten it to method_metric columns."
    df = pd.read_csv(path, header=[0, 1])
    cols = [c1 if c0.startswith('Unnamed') or c0 == 'info' else f'{c0}_{c1}'
            for c0, c1 in df.columns]
    cols[0] = id_name                                          # the leading unnamed column is the row id
    df.columns = cols
    return df


def _leave_one_out():
    "Per-kinase leave-one-out kNN accuracy for all three targets, computed live exactly as kd_06b does."
    import kd_06b_confidence_threshold as k6
    import kd_util
    rows = []
    for target in ['pspa', 'cddm', 'mlp_attr']:
        df, feat_col, target_col = kd_util.load_train(target, 'onehot')
        X, Y = df[feat_col].to_numpy(float), df[target_col].to_numpy(float)
        ids = df.iloc[:, 0].to_numpy()
        tax = kd_util.kinase_taxonomy(pd.Index(ids)).set_index('kinase').reindex(ids)
        P, nnd = k6.loo_retrieval(X, Y)                          # leave-one-out k=5 retrieval
        sp, ap, pear = kd_util.pssm_scores(Y, P, target_col)     # flank-only spearman / AP@5 / pearson
        rows.append(pd.DataFrame({
            'kd_ID': ids, 'target': target,
            'subfamily': tax['subfamily'].to_numpy(), 'family': tax['family'].to_numpy(),
            'group': tax['group'].to_numpy(),
            'spearman': sp, 'ap': ap, 'pearson': pear, 'nn_dist': nnd}))
    return pd.concat(rows, ignore_index=True).round(4)


def build_s5():
    model_feat = _melt_model_feature()
    selected = pd.read_excel(OUT / 'kd_model_selection.xlsx', 'selected_cell')
    loo = _leave_one_out()

    cutoff = pd.read_parquet(OUT / 'kd_confidence_threshold.parquet')
    by_kinase = _clean_validation(OUT / 'kd_validation_by_kinase.csv', 'kinase')
    by_species = _clean_validation(OUT / 'kd_validation_nonhuman.csv', 'uniprot')

    # taxonomy silhouette only (the paper shows silhouette; drop ARI/AMI per the figure)
    sil = pd.read_excel(OUT / 'cluster_validation.xlsx', 'overlap_PSPA_CDDM')
    sil = sil[['method', 'level', 'n', 'silhouette']]

    readme = [
        ('WORKBOOK', 'Supplementary Data S5 | Kinase-domain prediction model (Fig. 7, Supplementary '
                     'Fig. 2), testing whether a catalytic-domain sequence can predict a kinase substrate '
                     'PSSM. There is no held-out test set: every model x feature cell is scored by repeated '
                     "subfamily-grouped cross-validation over each target's full labeled set, with "
                     'hyperparameters fixed once by a separate grid search. kNN x one-hot (k = 5) is the '
                     'prespecified model, chosen for interpretable retrieval and the proximity distance it '
                     'provides rather than selected as the best-scoring cell of the grid.'),
        ('model_x_feature', 'Tidy model x feature CV scores per target (pspa/cddm/mlp_attr) x subset '
                            '(ST/TK/all) x metric (spearman/ap/pearson), value = median over kinases and '
                            'repeats. Behind Supplementary Fig. 2f, g.'),
        ('selected_cell', 'The selected cell (kNN x one-hot) per target x subset with median score and '
                          '95% CI for each metric.'),
        ('leave_one_out', 'Per-kinase leave-one-out (k = 5) prediction accuracy for all three targets '
                          '(PSPA, CDDM, MLP-attr): flank-only spearman / ap (AP@5) / pearson against the '
                          'measured PSSM, and nn_dist to the nearest other labeled kinase. This is the same '
                          'leave-one-out retrieval used to calibrate the proximity cutoff.'),
        ('proximity_cutoff', 'Per target (pspa/cddm/mlp_attr), one row each (Fig. 7b). threshold = the '
                             'applied proximity cutoff on nn_dist (the elbow of the PSPA leave-one-out '
                             'curve, 11.83, shared across all targets); threshold_lo/threshold_hi = its '
                             'bootstrap range; own_elbow = the target\'s own elbow (shown, not applied). '
                             'b_high / b_med = the nn_dist boundaries splitting the within-cutoff region '
                             'into high, intermediate and low proximity tiers (equal-distance thirds of '
                             'raw domain distance, not calibrated to accuracy). n_high / n_intermediate / '
                             'n_low = domains '
                             'in each tier; n_predictable = domains within the cutoff; n_query = all active '
                             'domains scored; drop_tk = whether TK domains were excluded for this target.'),
        ('external_val_by_kinase', 'Fig. 7d: external validation against non-human PSP orthologs, one row '
                                   'per unique human-mapped kinase (species averaged first to avoid '
                                   'pseudoreplication). Columns are {method}_{spearman|ap|pearson}.'),
        ('external_val_by_species', 'The same external validation before collapsing, one row per '
                                    'species-kinase pair, with nn_dist to the labeled human domain.'),
        ('taxonomy_silhouette', 'Supplementary Fig. 2d: cut-free silhouette of each motif space (PSPA / '
                                'CDDM / MLP-attr) against kinase group / family / subfamily, over the 293 '
                                'kinases shared by PSPA and CDDM (position 0 included). Higher = more '
                                'taxonomic separation.'),
    ]
    write_book('Supplementary_Data_S5_kinase_domain_model.xlsx',
               {'model_x_feature': model_feat, 'selected_cell': selected, 'leave_one_out': loo,
                'proximity_cutoff': cutoff, 'external_val_by_kinase': by_kinase,
                'external_val_by_species': by_species, 'taxonomy_silhouette': sil}, readme)


# ----------------------------------------------------------------------------------------------------
# S6 - predicted kinase domains (copy the kd_10 deliverable)
# ----------------------------------------------------------------------------------------------------
def build_s6():
    src = OUT / 'kd_predictions.xlsx'
    if not src.exists():
        sys.exit(f'{src} missing - run kd_10_prediction_excel.py first')
    SUPP.mkdir(exist_ok=True)
    dst = SUPP / 'Supplementary_Data_S6_predicted_kinase_domains.xlsx'
    shutil.copy2(src, dst)
    print(f'  copied {dst.name}  ({dst.stat().st_size / 1e6:.1f} MB; its own "description" sheet is the README)')


# ----------------------------------------------------------------------------------------------------
# S7 - pathway recovery (Fig. 8)
# ----------------------------------------------------------------------------------------------------
#: one top-pathway sheet per scoring method, as (sheet label, site-centric tag, method name in the
#: per-kinase agreement table). PSPA is the experimental reference the other two are compared against,
#: so it carries no agreement column of its own. One sheet per method rather than one long pooled sheet:
#: every scored kinase is listed, which is ~1,000 kinases across the three.
TOP_METHODS = [('PSPA', 'pspa_sc', None),
               ('CDDM', 'cddm_sc', 'CDDM freq'),
               ('MLP_attr', 'mlp_sc', 'CDDM MLP-attr')]
TOP_N_PATHWAYS = 10                                # the figure draws the first p4.TOP_N of these


def _top_pathways():
    """Per method, the top pathways of every scored kinase, as {sheet name: dataframe}.

    CDDM and MLP-attr rows also carry that kinase's AP@5 agreement with PSPA, so a pathway list can be
    read against how closely the kinase's motif matches the experimental reference. Ranks 1..p4.TOP_N
    of the eight Fig. 8a-h kinases in the CDDM sheet are the bars of that figure.
    """
    import pathway_04_cddm_examples as p4
    bg, N, pw, pw_u, pw_K, names, ann = p4.universe()
    info = kdata.load('kinase_info').drop_duplicates('kinase').set_index('kinase')
    kin_uni, kin_grp = info['uniprot'].to_dict(), info['group'].to_dict()
    agree = pd.read_csv(OUT / 'compare_vs_pspa_bygroup.csv')          # per-kinase agreement with PSPA
    ap5 = {(m, k): v for m, k, v in zip(agree.method, agree.kinase, agree.ap)}

    sheets = {}
    for label, tag, agree_method in TOP_METHODS:
        M = p4.ora_matrix(tag, bg, N, pw, pw_u, pw_K)
        spec = M.sub(M.mean(axis=1), axis=0)
        tested = set(spec.index)
        rows = []
        for kinase in sorted(spec.columns):
            ref_ids = ann.get(kin_uni.get(kinase), set()) & tested
            col = spec[kinase].sort_values(ascending=False)
            picked, seen = [], set()
            for r in col.index:                                        # figure dedups near-duplicate labels
                lab = p4.clip(names.get(r, r))
                if lab in seen:
                    continue
                seen.add(lab); picked.append(r)
                if len(picked) == TOP_N_PATHWAYS:
                    break
            a = None if agree_method is None else ap5.get((agree_method, kinase))
            for rank, r in enumerate(picked, 1):
                row = {'kinase': kinase, 'group': kin_grp.get(kinase), 'rank': rank,
                       'pathway': names.get(r, r), 'pathway_id': r,
                       'specificity': round(float(col[r]), 4),
                       'annotated_in_reactome': r in ref_ids,
                       'n_annotated_in_reactome': len(ref_ids)}
                if agree_method is not None:
                    row['ap_at5_vs_PSPA'] = None if a is None else round(float(a), 4)
                rows.append(row)
        sheets[f'top_pathways_{label}'] = pd.DataFrame(rows)
    return sheets


def build_s7():
    recovery = pd.read_parquet(OUT / 'pathway_localora_eval.parquet')
    # the paper uses the site-centric selection only, so that column is constant and is dropped
    recovery = recovery[recovery.selection == 'sitecentric'].drop(columns='selection').round(4)
    perk = pd.read_parquet(OUT / 'pathway_cddm_perkinase_auroc.parquet').reset_index().round(4)
    perk = perk.rename(columns={perk.columns[0]: 'kinase'})
    sheets = {'kinome_wide_recovery': recovery, 'per_kinase_recovery_cddm': perk}

    readme = [
        ('WORKBOOK', 'Supplementary Data S7 | Pathway recovery (Fig. 8). Score every unique human '
                     'phosphosite with PSPA and CDDM PSSMs and with the CDDM-seq MLP, assign a site to a '
                     'kinase when it ranks in the top 3 and reaches the 95th percentile, then test Reactome '
                     'pathway over-representation against the scored phosphoproteome. Pathway specificity = '
                     "the -log10(P) of a pathway for a kinase minus that kinase's kinome-wide mean."),
        ('kinome_wide_recovery', 'Kinome-wide recovery per method x ranking x split (S/T, TK, all): AUROC '
                                 "of placing a kinase's Reactome-annotated pathways above the rest, its "
                                 'permutation null (auroc_null) and the null-corrected signal (Fig. 8i-k). '
                                 'The analysis reported in the paper uses ranking = specificity.'),
        ('per_kinase_recovery_cddm', 'Per-kinase pathway-recovery AUROC for CDDM, with kinase group '
                                     '(kinases with at least 10 Reactome-annotated pathways).'),
    ]
    try:
        import pathway_04_cddm_examples as p4
        sheets.update(_top_pathways())
        shared = ("specificity = the pathway-specificity score, so rank 1 is that kinase's most specific "
                  'enriched pathway; annotated_in_reactome = whether the pathway is annotated to that '
                  'kinase in Reactome; n_annotated_in_reactome = how many of the tested pathways are '
                  'annotated to that kinase at all, which separates a kinase Reactome does not annotate '
                  'from one whose annotated pathways were simply not recovered.')
        for label, _, agree_method in TOP_METHODS:
            name = f'top_pathways_{label}'
            txt = (f'The top {TOP_N_PATHWAYS} pathways for each of the '
                   f'{sheets[name].kinase.nunique()} kinases scored with '
                   f'{label.replace("_", "-")} PSSMs. {shared}')
            if agree_method is not None:
                txt += (" ap_at5_vs_PSPA = that kinase's AP@5 agreement with PSPA, so a pathway list can "
                        'be read against how closely the motif matches the experimental reference; blank '
                        'where the kinase is not measured by both methods.')
            if label == 'CDDM':
                txt += (f' Fig. 8a-h shows ranks 1 to {p4.TOP_N} for AKT1, ATM, CAMK2A, ERK2, PAK4, BUB1, '
                        'SYK and TAK1, with annotated pathways drawn as red bars and the rest as gray.')
            readme.append((name, txt))
    except Exception as e:
        print(f'  (representative_pathways skipped: {type(e).__name__}: {e})')

    write_book('Supplementary_Data_S7_pathway_recovery.xlsx', sheets, readme)


BUILDERS = {'S1': build_s1, 'S2': build_s2, 'S3': build_s3, 'S4': build_s4,
            'S5': build_s5, 'S6': build_s6, 'S7': build_s7}


def main():
    want = [a.upper() for a in sys.argv[1:]] or list(BUILDERS)
    unknown = [w for w in want if w not in BUILDERS]
    if unknown:
        sys.exit(f'unknown workbook(s): {unknown}; choose from {list(BUILDERS)}')
    print('supplement_table:', SUPP)
    for w in want:
        print(f'\n== {w} ==')
        BUILDERS[w]()


if __name__ == '__main__':
    main()
