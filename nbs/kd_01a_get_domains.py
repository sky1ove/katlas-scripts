"""kd_01a · Extract every kinase domain from UniProt, across species.

The kinome-tree table only covers human kinases whose domain boundaries are already curated.
This widens that: four UniProt advanced searches (reviewed entries) are pooled —

    domain: "Protein kinase"          uniprotkb_ft_domain_Protein_kinase_AND_*.xlsx
    domain: "PI3K PI4K catalytic"     uniprotkb_ft_domain_PI3K_PI4K_catalytic_*.xlsx
    domain: "Histidine kinase"        uniprotkb_ft_domain_Histidine_kinase_AN_*.xlsx
    region: "Kinase domain"           uniprotkb_ft_region_Kinase_domain_*.xlsx

— and each entry's `Domain [FT]` (or `Region`, for the fourth) is parsed for the start/end of
every kinase-like annotation. A protein with two catalytic domains (JAK1, and the kinases whose
second domain is histidine-kinase) yields two rows, keyed `<uniprot>_<entry name>_KD<n>`.

`AGC-kinase C-terminal` annotations match the regex but are not catalytic domains, so they are
dropped.

NOTE the original notebook also re-downloaded every full sequence from UniProt to cross-check;
the download truncates long proteins (titin), so it immediately reverted to the sequence already
in the spreadsheet. That block is dead and is not ported.

Inputs   raw/uniprotkb_ft_*.xlsx (4 UniProt exports), raw/idmapping_kinase_info_2025_05_27.xlsx,
         raw/uniprot_human_keyword_kinase.xlsx, kdata.load('kinase_info')
Outputs  out/kd_uniprot.parquet   one row per kinase domain

Run:  python nbs/kd_01a_get_domains.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from paths import OUT, RAW

import kdata

DOMAIN_FILES = ['uniprotkb_ft_domain_Protein_kinase_AND_2025_05_25.xlsx',
                'uniprotkb_ft_domain_PI3K_PI4K_catalytic_2025_05_25.xlsx',
                'uniprotkb_ft_domain_Histidine_kinase_AN_2025_05_25.xlsx']
REGION_FILE = 'uniprotkb_ft_region_Kinase_domain_2025_05_26.xlsx'

#: a DOMAIN/REGION annotation whose note mentions a kinase, with optional evidence
DOMAIN_RE = re.compile(
    r'(?:REGION|DOMAIN) [<>]?(\d+)\.\.[<>]?(\d+); /note="'
    r'([^"]*?(?:kinase|PI3K/PI4K catalytic)[^"]*?)"(?:; /evidence="([^"]*?)")?',
    flags=re.IGNORECASE)

DROP_NOTES = ['AGC-kinase C-terminal']      # matches the regex but is not a catalytic domain

COLUMNS = ['kd_ID', 'Uniprot', 'Entry Name', 'Protein names', 'Gene Names',
           'Gene Names (primary)', 'Organism', 'kd_note', 'kd_evidence', 'kd_start', 'kd_end',
           'kd_seq', 'Domain [FT]', 'Domain [CC]', 'Region', 'Motif', 'Protein families',
           'Reactome', 'ComplexPortal', 'Subcellular location [CC]',
           'Gene Ontology (biological process)', 'Tissue specificity', 'Interacts with',
           'Subunit structure', 'Function [CC]', 'Activity regulation', 'full_seq']
RENAME = {'KD_ID': 'kd_ID', 'domain_note': 'kd_note', 'domain_evidence': 'kd_evidence',
          'domain_start': 'kd_start', 'domain_end': 'kd_end', 'domain_seq': 'kd_seq'}


def extract_domains(text):
    "[(note, start, end, evidence), ...] for every kinase-like annotation in one UniProt field."
    return [[note.strip(), int(start), int(end), evidence or 'nan']
            for start, end, note, evidence in DOMAIN_RE.findall(text)]


def load_queries():
    "The four UniProt exports, with the domain annotations parsed into `kd_info`."
    kd = pd.concat([pd.read_excel(RAW / f) for f in DOMAIN_FILES])
    kd = kd.drop_duplicates('Uniprot').reset_index(drop=True)
    kd['kd_info'] = kd['Domain [FT]'].apply(extract_domains)

    # the fourth search has no Domain [FT] - its boundaries live in Region
    kd4 = pd.read_excel(RAW / REGION_FILE)
    kd4['kd_info'] = kd4['Region'].apply(extract_domains)

    print(f'domain queries: {len(kd)} entries | region query: {len(kd4)} entries')
    print('  domains per entry:', dict(kd['kd_info'].str.len().value_counts().sort_index()))
    return pd.concat([kd, kd4])


def only_in(a, b, col_a, col_b=None):
    "Values of `col_a` in `a` that are absent from `b[col_b]` (katlas.utils.get_diff is gone)."
    return set(a[col_a].dropna()) - set(b[col_b or col_a].dropna())


def compare_to_kinome(kd_human):
    "Report how the domain query overlaps the kinome tree and the keyword query."
    info = kdata.load('kinase_info')
    print(f'vs kinome tree: {len(only_in(kd_human, info, "Uniprot", "uniprot"))} only in the '
          f'domain query, {len(only_in(info, kd_human, "uniprot", "Uniprot"))} only on the tree '
          '(no clear domain boundary)')

    keyword = pd.read_excel(RAW / 'uniprot_human_keyword_kinase.xlsx').rename(
        columns={'Entry': 'Uniprot'})
    print(f'vs keyword query: {len(only_in(kd_human, keyword, "Uniprot"))} only in the domain '
          "query (inactive kinases like PLK5, and NAGS's amino-acid kinase domain), "
          f'{len(only_in(keyword, kd_human, "Uniprot"))} only in the keyword query '
          '(mostly non-S/T/Y kinases)')


def build(kd_all):
    "One row per domain, with its sequence sliced out of the full protein."
    df = kd_all.explode('kd_info', ignore_index=True)
    df[['domain_note', 'domain_start', 'domain_end', 'domain_evidence']] = \
        df.kd_info.apply(pd.Series)
    df = df.drop(columns=['kd_info'])
    df = df[~df.domain_note.isin(DROP_NOTES)].reset_index(drop=True)
    print('  domain notes:', dict(df.domain_note.value_counts().head(6)))

    df = df.rename(columns={'Sequence': 'full_seq'})
    df['domain_seq'] = df.apply(lambda r: r['full_seq'][r['domain_start'] - 1:r['domain_end']],
                                axis=1)

    # The spreadsheet's Sequence column is capped at Excel's 32,767-character cell limit, so a
    # domain past that point slices to nothing. Only titin hits this (mouse A2ASS6's domain is at
    # 33,040-33,294); flag it rather than emitting a silent empty sequence.
    short = df[df.domain_seq.str.len() < (df.domain_end - df.domain_start + 1)]
    if len(short):
        print(f'\n  WARNING: {len(short)} domain(s) lie past the end of the stored full sequence '
              '(Excel truncates at 32,767 chars) - their kd_seq is incomplete:')
        for _, r in short.iterrows():
            print(f'    {r.Uniprot} {r["Entry Name"]}  domain {r.domain_start}-{r.domain_end}, '
                  f'full_seq is only {len(r.full_seq):,} aa')

    df = df.sort_values(['Uniprot', 'domain_start']).reset_index(drop=True)
    # KD1 / KD2 / ... by order along the protein, so JAK1's two domains stay distinguishable
    # (cumcount, not duplicated: a >=3-domain protein must get KD1/KD2/KD3, never a repeated KD2)
    df['KD_ID'] = (df['Uniprot'] + '_' + df['Entry Name'] + '_KD'
                   + (df.groupby('Uniprot').cumcount() + 1).astype(str))
    return df.rename(columns=RENAME)[COLUMNS]


def report_duplicates(df):
    "Identical domain sequences, within human and across the top species."
    def dups(d):
        dup = d[d.kd_seq.duplicated(keep=False)]
        return dup.groupby('kd_seq').agg({'kd_ID': lambda x: ','.join(x)}).reset_index()

    human = df[df.Organism == 'Homo sapiens (Human)']
    print(f'\nduplicate domain sequences within human: {len(dups(human))}')
    for _, r in dups(human).iterrows():
        print('  ', r.kd_ID)

    # tuple, not list: value_counts needs a hashable key
    per_uniprot = df.groupby('Uniprot').kd_note.agg(tuple).value_counts()
    print('\ndomain combinations per protein:')
    for combo, n in per_uniprot.head(5).items():
        print(f'  {n:5}  {" + ".join(combo)}')


def main():
    OUT.mkdir(exist_ok=True)

    kd_all = load_queries()
    compare_to_kinome(kd_all[kd_all.Organism.str.contains('Homo')])

    df = build(kd_all)
    print(f'\nkinase domains: {df.shape} | proteins: {df.Uniprot.nunique()} '
          f'| species: {df.Organism.nunique()}')
    report_duplicates(df)

    out = OUT / 'kd_uniprot.parquet'
    df.to_parquet(out, index=False)
    print('\nwrote', out)


if __name__ == '__main__':
    main()
