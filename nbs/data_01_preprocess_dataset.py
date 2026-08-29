"""data_01 · Build the unified kinase-substrate (KS) dataset.

Collects kinase-substrate phosphorylation pairs from PhosphoSitePlus (PSP), Sugiyama,
KiNET (EPSD + iPTMNet), Phospho.ELM, SIGNOR and GPS 6.0. For every source it

1. maps kinase names / accessions to UniProt IDs,
2. maps substrate accessions to UniProt IDs + full protein sequences (human only),
3. validates that the residue named by the site actually sits at that position,

then concatenates everything, groups by `kin_sub_site`, phosphorylates the substrate
sequences (lowercase s/t/y at known phosphosites) and extracts +-20 aa site sequences.

The same treatment is applied to the human phosphoproteome (PSP + Ochoa).

Finally the grouped table is annotated with the kinase columns (from `kinase_info`) and
`num_kin` (distinct kinases per substrate site) and saved as the `ks_dataset` dataset -
the table `kdata.ks_dataset(thr=...)` serves and everything downstream builds on.

Inputs   raw/  (downloaded source tables + UniProt idmapping exports, read-only)
         kdata: kinase_info, kinase_uniprot, combine_site_psp_ochoa
Outputs  out/  source_<name>.parquet, combine_source.parquet,
               combine_source_grouped.parquet, human_phosphoproteome.parquet,
               phosphoseq_map.csv
         katlas_datasets/CDDM/ks_datasets.parquet

Source links
  Sugiyama  https://www.nature.com/articles/s41598-019-46385-4 (table S2)
  PSP       https://www.phosphosite.org/staticDownloads (Kinase_Substrate_Dataset.txt)
  KiNET     https://kinet.kinametrix.com/ (full interaction dataset)
  ELM       http://phospho.elm.eu.org/dataset.html
  SIGNOR    https://signor.uniroma2.it/ (latest release, mechanism = phosphorylation)
  GPS 6.0   https://academic.oup.com/nar/article/51/W1/W243/7157529 (table S5)

Run:  python nbs/data_01_preprocess_dataset.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from paths import RAW, OUT

import kdata
from katlas.utils import extract_site_seq, phosphorylate_seq_df, validate_site_df

# ---------------------------------------------------------------- shared helpers


def map_substrate(idmapping_fname, ori_df, sub_col, remove_nonhuman=True):
    "Merge a UniProt idmapping export onto `ori_df`, attaching substrate accession/genes/sequence."
    substrate_id = pd.read_excel(idmapping_fname)
    substrate_id = substrate_id.drop_duplicates('From')
    substrate_id = substrate_id[['From', 'Entry', 'Gene Names', 'Sequence', 'Organism']]
    ori_df = ori_df.copy()

    # prevent name conflict
    if sub_col == 'substrate_uniprot':
        sub_col = 'substrate_uniprot_tmp'
        ori_df = ori_df.rename(columns={'substrate_uniprot': 'substrate_uniprot_tmp'})

    substrate_id.columns = [sub_col, 'substrate_uniprot', 'substrate_genes',
                            'substrate_sequence', 'substrate_species']

    print('  shape before substrate mapping:', ori_df.shape)
    ori_df = ori_df.merge(substrate_id, on=sub_col)

    if remove_nonhuman:
        ori_df = ori_df[ori_df.substrate_species == 'Homo sapiens (Human)']

    ori_df = ori_df.drop(columns=[sub_col])
    ori_df = ori_df.dropna(subset=['substrate_sequence'])
    print('  shape after substrate mapping :', ori_df.shape)
    return ori_df.reset_index(drop=True)


def build_gene2uniprot():
    "Gene-name -> UniProt Entry dict, plus the set of gene names shared by >1 kinase."
    kinase_id = kdata.load('kinase_uniprot')
    kinase_id['Gene Names'] = kinase_id['Gene Names'].str.split(' ')
    kinase_id = kinase_id.explode('Gene Names')
    dup_name = set(kinase_id[kinase_id['Gene Names'].duplicated(keep=False)]['Gene Names'])
    gene2uniprot = kinase_id.set_index('Gene Names')['Entry'].to_dict()
    return gene2uniprot, dup_name


def filter_valid_sites(df, site_col='site', seq_col='substrate_sequence'):
    "Keep rows whose site residue matches the residue at that position in the protein sequence."
    site_match = validate_site_df(df, site_col, seq_col)
    print('  site validation:', dict(site_match.value_counts()))
    df = df[site_match == 1].copy()
    print('  shape after site validation   :', df.shape)
    return df


# ---------------------------------------------------------------- per-source loaders


def load_sugiyama():
    "Sugiyama & Douglass large-scale in-vitro screen (table S2)."
    print('\n== Sugiyama ==')
    df = pd.read_csv(RAW / 'Large_scale_S2.csv').iloc[:, :-2]

    # kinase names were mapped to UniProt IDs by hand -> LS_info2.csv
    kinase_id = pd.read_csv(RAW / 'LS_info2.csv').iloc[:, :3]
    df = df.merge(kinase_id)

    df = map_substrate(RAW / 'idmapping_2025_03_02.xlsx', df, 'Substrate_uniprot')
    df = df.rename(columns={'Position': 'site'})
    df = filter_valid_sites(df)
    df['source'] = 'Sugiyama'
    return df


def load_psp():
    "PhosphoSitePlus kinase-substrate dataset, human-human pairs only."
    print('\n== PSP ==')
    psp = pd.read_csv(RAW / 'Kinase_Substrate_Dataset_final.csv')
    psp = psp[psp.KIN_ORGANISM == 'human']
    psp = psp[psp.SUB_ORGANISM == 'human'].reset_index(drop=True)
    psp = psp[['KIN_ACC_ID', 'kinase_paper', 'GENE', 'SUB_ACC_ID', 'SUB_GENE',
               'SUB_MOD_RSD', 'substrate']]

    psp = map_substrate(RAW / 'idmapping_2025_03_02_psp.xlsx', psp, 'SUB_ACC_ID')

    kinase_id = pd.read_excel(RAW / 'idmapping_2025_03_02_psp_kinase.xlsx')[
        ['From', 'Entry', 'Gene Names']]
    kinase_id.columns = ['KIN_ACC_ID', 'kinase_uniprot', 'kinase_genes']
    psp['KIN_ACC_ID'] = psp['KIN_ACC_ID'].replace('A9UF07', 'P00519')  # ABL1
    psp = psp.merge(kinase_id)

    psp = psp.rename(columns={'SUB_MOD_RSD': 'site'})
    psp = psp[['kinase_uniprot', 'kinase_genes', 'kinase_paper', 'substrate_uniprot',
               'substrate_genes', 'site', 'substrate_sequence', 'substrate',
               'substrate_species']]
    psp = psp.dropna(subset='substrate_sequence').reset_index(drop=True)

    # CSNK2B / PRKAB1 / PRKAG2 are regulatory subunits, not kinases themselves -> drop
    kinase_uniprot = kdata.load('kinase_uniprot')
    psp = psp[psp.kinase_uniprot.isin(kinase_uniprot.Entry)].copy()

    # PSP ships its own +-7 site sequence; keep only rows where it agrees with UniProt
    psp['site_seq'] = extract_site_seq(psp, seq_col='substrate_sequence', site_col='site', n=7)
    match = psp['site_seq'] == psp['substrate'].str.upper()
    print('  PSP site_seq vs UniProt:', dict(match.value_counts()))
    psp = psp[match]
    psp = psp[psp['site_seq'].str[7].str.upper().isin(list('STY'))]

    psp = filter_valid_sites(psp)
    psp['source'] = 'PSP'
    return psp


def load_kinet():
    "KiNET (integration of PSP, iPTMNet, EPSD); PSP rows dropped since PSP is loaded separately."
    print('\n== KiNET (EPSD + iPTMNet) ==')
    df = pd.read_csv(RAW / 'ksi_source_full_dataset.csv')
    df = df.dropna(subset='Kinase')
    df = df[df['Source Database'] != 'PhosphoSitePlus']

    kinase_uniprot = kdata.load('kinase_uniprot')
    df = df[df.Kinase.isin(kinase_uniprot.Entry)]

    df = map_substrate(RAW / 'idmapping_2025_03_02_KiNET_substrate.xlsx', df, 'Substrate')
    df = df[['Kinase', 'Kinase Name', 'substrate_uniprot', 'substrate_genes', 'Site',
             'Source Database', 'Evidence', 'substrate_sequence']]
    df.columns = ['kinase_uniprot', 'Kinase Name', 'substrate_uniprot', 'substrate_genes',
                  'site', 'source', 'evidence', 'substrate_sequence']

    df = filter_valid_sites(df)
    print('  per source:', dict(df.source.value_counts()))
    return df


def load_elm():
    "Phospho.ELM, human rows with a non-blank kinase."
    print('\n== Phospho.ELM ==')
    elm = pd.read_csv(RAW / 'phosphoELM.csv')
    elm.kinase = elm.kinase.str.upper()

    gene2uniprot, dup_name = build_gene2uniprot()
    # these three are ambiguous by gene name but have an unambiguous intended kinase
    for gene, acc in [('PAK1', 'Q13153'), ('PASK', 'Q96RG2'), ('PRKACA', 'P17612')]:
        dup_name.discard(gene)
        gene2uniprot[gene] = acc

    elm = elm[~elm.kinase.isin(dup_name)].copy()

    ids = pd.read_csv(RAW / 'elm_kinase_id.csv').set_index('kinase')['kinase_gene'].to_dict()
    elm['kinase_genes'] = elm.kinase.map(ids).fillna(elm.kinase)
    # for kinase families we only keep the first two members
    elm['kinase_genes'] = elm.kinase_genes.str.split(' ').str[:2]
    elm = elm.explode('kinase_genes')
    elm = elm[~elm.kinase_genes.isin(dup_name)]

    elm['kinase_uniprot'] = elm.kinase_genes.map(gene2uniprot)
    elm = elm.dropna(subset='kinase_uniprot')

    # two "substrates" are actually kinases whose ENSP could not be mapped by UniProt
    ensp = {'ENSP00000328213': 'P06239', 'ENSP00000261937': 'P35916'}
    elm.substrate_uniprot = elm.substrate_uniprot.map(ensp).fillna(elm.substrate_uniprot)

    elm = map_substrate(RAW / 'idmapping_2025_03_12_elm.xlsx', elm, 'substrate_uniprot')

    elm['site'] = elm['acceptor'] + elm['position'].astype(str)
    elm = filter_valid_sites(elm)

    elm = elm[['kinase', 'kinase_uniprot', 'kinase_genes', 'substrate_uniprot',
               'substrate_genes', 'site', 'LTP_HTP', 'species', 'substrate_sequence']]
    elm['source'] = 'ELM'
    return elm


def load_signor():
    "SIGNOR phosphorylation reactions; complexes / fusions / families resolved to genes."
    print('\n== SIGNOR ==')
    sig = pd.read_excel(RAW / 'signor_phosphorylation.xlsx')

    # ENTITYA is a complex -> hand-curated complex -> kinase gene table
    comp = sig[sig.TYPEA == 'complex'].copy()
    comp_id = pd.read_csv(RAW / 'sig_complex_label.csv').set_index('ENTITYA')['kinase_gene']
    comp['kinase_gene'] = comp.ENTITYA.map(comp_id)
    comp = comp.dropna(subset='kinase_gene')

    # fusion proteins -> the kinase half
    fus = sig[sig.TYPEA == 'fusion protein'].copy()
    fus['kinase_gene'] = fus.ENTITYA.map({'BCR-ABL': 'ABL1', 'EML4-ALK': 'ALK'})

    # protein families -> first two members
    fam = sig[sig.TYPEA == 'proteinfamily'].copy()
    fam_id = pd.read_csv(RAW / 'sig_fam_label.csv').set_index('ENTITYA')['kinase_gene']
    fam['kinase_gene'] = fam.ENTITYA.map(fam_id)
    fam = fam.dropna(subset='kinase_gene')
    fam['kinase_gene'] = fam.kinase_gene.str.split(' ').str[:2]
    fam = fam.explode('kinase_gene')

    pro = sig[sig.TYPEA == 'protein'].copy()
    pro['kinase_gene'] = pro['ENTITYA']

    df = pd.concat([pro, comp, fus, fam])

    gene2uniprot, dup_name = build_gene2uniprot()
    df = df[~df.kinase_gene.isin(dup_name)].copy()
    df['kinase_uniprot'] = df.kinase_gene.str.upper().map(gene2uniprot)
    df = df.dropna(subset='kinase_uniprot')

    # SIGNOR-internal IDs have no traceable protein sequence
    df = df.dropna(subset='IDB')
    df = df[~df.IDB.str.contains('SIGNOR')]
    df = map_substrate(RAW / 'idmapping_2025_03_12_signor.xlsx', df, 'IDB')

    df.RESIDUE = df.RESIDUE.str.split(';')
    df = df.explode('RESIDUE')
    df['acceptor'] = df['RESIDUE'].str[:3].map({'Ser': 'S', 'Thr': 'T', 'Tyr': 'Y'})
    df['position'] = df['RESIDUE'].str[3:]
    df['site'] = df['acceptor'] + df['position']
    df = df.dropna(subset='site')
    df = filter_valid_sites(df)

    df = df[['kinase_uniprot', 'kinase_gene', 'ENTITYA', 'TYPEA', 'substrate_uniprot',
             'substrate_genes', 'site', 'substrate_sequence']].copy()
    df['source'] = 'SIGNOR'
    return df


def load_gps():
    "GPS 6.0 supplementary table S5, human rows, PSP-derived rows dropped."
    print('\n== GPS 6.0 ==')
    gps = pd.read_csv(RAW / 'GPS6_tableS5.csv')
    gps = gps[gps.source != 'PhosphositePlus']
    gps = gps[gps.species == 'Homo sapiens']
    gps = gps[~gps.gene.str.contains('family')]

    gene2uniprot, dup_name = build_gene2uniprot()
    gps = gps[~gps.gene.isin(dup_name)].copy()
    gps['kinase_uniprot'] = gps.gene.str.upper().map(gene2uniprot)
    gps = gps.dropna(subset='kinase_uniprot')

    gps = map_substrate(RAW / 'idmapping_2025_03_12_GPS.xlsx', gps, 'uniprot')

    gps['site'] = gps['code'] + gps['position'].astype(int).astype(str)
    gps = filter_valid_sites(gps)

    gps = gps.rename(columns={'source': 'GPS_source'})
    gps['source'] = 'GPS6'
    gps = gps[['kinase_uniprot', 'gene', 'substrate_uniprot', 'substrate_genes', 'site',
               'substrate_sequence', 'GPS_source', 'source']].copy()
    return gps


# ---------------------------------------------------------------- combine


COMMON_COLS = ['kinase_uniprot', 'substrate_uniprot', 'site', 'kin_sub_site', 'source',
               'substrate_genes', 'substrate_sequence']


def add_key(df, name=''):
    "Add the kinase-substrate-site key and drop duplicates on it (within one source)."
    df = df.copy()
    before = df.shape
    df['kin_sub_site'] = df['kinase_uniprot'] + '_' + df['substrate_uniprot'] + '_' + df['site']
    df = df.drop_duplicates(subset='kin_sub_site')
    print(f'  {name:<10} {before} -> {df.shape}')
    return df


def combine_sources(sources):
    "Concatenate per-source frames (deduplicated within source) and group by kin_sub_site."
    print('\n== Combine sources ==')
    dfs = [add_key(df, name) for name, df in sources.items()]
    df_all = pd.concat(dfs, ignore_index=True)[COMMON_COLS].copy()
    print('  concatenated:', df_all.shape)
    print('  per source  :', dict(df_all.source.value_counts()))

    df_grouped = df_all.groupby('kin_sub_site').agg({
        'kinase_uniprot': 'first',
        'substrate_uniprot': 'first',
        'site': 'first',
        'source': '|'.join,
        'substrate_genes': 'first',
        'substrate_sequence': 'first',
    }).reset_index()
    print('  grouped     :', df_grouped.shape)
    return df_all, df_grouped


# ---------------------------------------------------------------- human phosphoproteome


def build_human_phosphoproteome():
    "PSP + Ochoa phosphosites, mapped to UniProt sequences and site-validated."
    print('\n== Human phosphoproteome (PSP + Ochoa) ==')
    human = kdata.load('combine_site_psp_ochoa')
    human = map_substrate(RAW / 'idmapping_2025_03_20_human_phosphoproteome.xlsx',
                          human, 'uniprot')
    human = filter_valid_sites(human)

    human['sub_site'] = human['substrate_uniprot'] + '_' + human['site']
    human = human.drop_duplicates(subset='sub_site')
    print('  after site dedup:', human.shape)

    return human[['substrate_uniprot', 'substrate_genes', 'site', 'source',
                  'AM_pathogenicity', 'substrate_sequence', 'substrate_species',
                  'sub_site']].copy()


# ---------------------------------------------------------------- sequences


def build_phosphoseq_map(human, df_grouped):
    "Phosphorylate each substrate sequence using every known site across both tables."
    print('\n== Phosphorylate substrate sequences ==')
    cols = ['substrate_uniprot', 'site', 'substrate_sequence']
    comb = pd.concat([human[cols], df_grouped[cols]])
    comb['sub_site'] = comb['substrate_uniprot'] + '_' + comb['site']
    comb = comb.drop_duplicates('sub_site')
    print('  unique substrate sites:', comb.shape)

    seq = phosphorylate_seq_df(comb)
    print('  phosphorylated proteins:', seq.shape)
    return seq


def add_site_seq(df, seq_map, n=20):
    "Attach the phosphorylated full sequence, the +-n site sequence, position and sub_site."
    df = df.copy()
    df['substrate_phosphoseq'] = df.substrate_uniprot.map(seq_map)
    missing = df['substrate_phosphoseq'].isna()
    if missing.any():
        # extract_site_seq would die on NaN with an opaque TypeError, so stop here instead
        raise ValueError(
            f'{missing.sum()} rows have no phosphoseq, e.g. '
            f'{sorted(df.loc[missing, "substrate_uniprot"].unique())[:5]}. '
            'build_phosphoseq_map() should cover every substrate in both tables.')
    df['position'] = df['site'].str[1:].astype(int)
    df['site_seq'] = extract_site_seq(df, seq_col='substrate_phosphoseq', site_col='site', n=n)
    df['sub_site'] = df['substrate_uniprot'] + '_' + df['site']
    return df


# column layout of the `ks_dataset` dataset, in order
KS_DATASET_COLS = [
    'kin_sub_site', 'kinase_uniprot', 'substrate_uniprot', 'site', 'source',
    'substrate_genes', 'substrate_phosphoseq', 'position', 'site_seq', 'sub_site',
    'substrate_sequence', 'kinase_on_tree', 'kinase_genes', 'kinase_group',
    'kinase_family', 'kinase_subfamily', 'kinase_pspa_big', 'kinase_pspa_small',
    'kinase_coral_ID', 'kinase_protein', 'num_kin',
]

#: ks_dataset kinase column -> the kinase_info column it is mapped from
KINASE_INFO_MAP = {
    'kinase_protein': 'kinase',
    'kinase_group': 'group',
    'kinase_family': 'family',
    'kinase_subfamily': 'subfamily',
    'kinase_coral_ID': 'ID_coral',
    'kinase_pspa_big': 'pspa_category_big',
    'kinase_pspa_small': 'pspa_category_small',
}


def build_ks_dataset(df_grouped):
    """Annotate the grouped KS pairs into the `ks_dataset` dataset.

    Adds the kinase columns (mapped from `kinase_info` by UniProt accession, isoform suffix
    stripped) and `num_kin`, the number of distinct kinases targeting each substrate site -
    which is what `kdata.ks_dataset(thr=...)` filters on.
    """
    print('\n== Annotate ks_dataset ==')
    df = df_grouped.copy()
    info = (kdata.load('kinase_info').sort_values('kinase')
            .drop_duplicates('uniprot').set_index('uniprot'))
    clean = df['kinase_uniprot'].str.split('-').str[0]

    df['kinase_on_tree'] = clean.isin(info.index).astype(int)
    df['kinase_genes'] = clean.map(kdata.load('kinase_uniprot').set_index('Entry')['Gene Names'])
    for out_col, info_col in KINASE_INFO_MAP.items():
        df[out_col] = clean.map(info[info_col])

    df['num_kin'] = df.groupby('sub_site')['kinase_uniprot'].transform('nunique')

    print('  kinases on the kinome tree:', int(df.kinase_on_tree.sum()), '/', len(df), 'pairs')
    print('  num_kin: min', int(df.num_kin.min()), 'max', int(df.num_kin.max()))
    return df[KS_DATASET_COLS]


# ---------------------------------------------------------------- main


def main():
    OUT.mkdir(exist_ok=True)

    kinet = load_kinet()
    sources = {
        'GPS6': load_gps(),
        'SIGNOR': load_signor(),
        'ELM': load_elm(),
        # KiNET is split so that a pair seen in both sub-databases keeps both source tags
        'iPTMNet': kinet[kinet.source == 'iPTMNet'],
        'EPSD': kinet[kinet.source == 'EPSD'],
        'PSP': load_psp(),
        'Sugiyama': load_sugiyama(),
    }
    for name, df in sources.items():
        df.to_parquet(OUT / f'source_{name}.parquet', index=False)

    df_all, df_grouped = combine_sources(sources)
    df_all.to_parquet(OUT / 'combine_source.parquet', index=False)

    human = build_human_phosphoproteome()

    seq = build_phosphoseq_map(human, df_grouped)
    seq.to_csv(OUT / 'phosphoseq_map.csv', index=False)

    seq_map = seq.set_index('substrate_uniprot')['phosphoseq']
    print('\n== Extract +-20 site sequences ==')
    human = add_site_seq(human, seq_map)
    df_grouped = add_site_seq(df_grouped, seq_map)

    human.to_parquet(OUT / 'human_phosphoproteome.parquet', index=False)
    df_grouped.to_parquet(OUT / 'combine_source_grouped.parquet', index=False)

    kdata.save('ks_dataset', build_ks_dataset(df_grouped), index=False)

    print('\nwrote')
    for f in ['combine_source.parquet', 'combine_source_grouped.parquet',
              'human_phosphoproteome.parquet', 'phosphoseq_map.csv']:
        print(' ', OUT / f)


if __name__ == '__main__':
    main()
