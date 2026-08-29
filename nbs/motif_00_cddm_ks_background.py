"""motif_00 · CDDM / KS background amino-acid frequencies for log-odds scoring.

Builds background PSSMs - the aa frequency at every flanking position, pooled over all
sites regardless of kinase - for two site pools:

  ks     all unique KS-dataset sites  (`kdata.load('ks_unique')`)
  human  the human phosphoproteome    (`kdata.load('human_site')`)

each split by phosphoacceptor (S, T, Y, ST, STY) and computed twice: case-preserving and
all-uppercase. Rows are named `<pool>_<acceptor>[_upper]`, e.g. `ks_STY`, `human_Y_upper`.
The CDDM log-odds (`cddm_LO`) divide the CDDM frequencies by this background, but `motif_01` is
their single producer — run this first (it must exist before `motif_01`), then `motif_01`.

Sites are deduplicated on the all-uppercase site sequence first, so sequences that differ
only in phospho-priming annotation are counted once.

Overwrites `ks_background` only; the CDDM log-odds that divide by it are derived by `motif_01`
(its single producer). The version it replaces is kept in `katlas_datasets/_archive/`.

Inputs   kdata.load('ks_unique'), kdata.load('human_site')
Outputs  katlas_datasets/CDDM/ks_background.parquet
         fig/background_ks.pdf, fig/background_human.pdf, fig/human_y_site.svg

Run:  python nbs/motif_00_cddm_ks_background.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from paths import FIG

import kdata
from katlas.plot import plot_logo_heatmap, plot_logos
from katlas.pssm import flatten_pssm, get_prob
from kplot.utils import save_pdf, save_svg

ACCEPTORS = ['S', 'T', 'Y', 'ST', 'STY']


def dedup_upper(df):
    "Drop sites whose all-uppercase site sequence has already been seen."
    df = df.copy()
    df['site_seq_upper'] = df['site_seq'].str.upper()
    before = len(df)
    df = df.drop_duplicates(subset='site_seq_upper').reset_index(drop=True)
    print(f'  {before} -> {len(df)} sites after uppercase dedup')
    return df


def get_bg_dict(df, acceptor, seq_col='site_seq'):
    "Flat background PSSM over the sites whose acceptor is in `acceptor`."
    site = df[df['acceptor'].isin(list(acceptor))]
    return flatten_pssm(get_prob(site, seq_col), True)


def get_site_cnt(df, acceptor):
    return int(df['acceptor'].isin(list(acceptor)).sum())


def build_background(df, prefix):
    "Case-preserving + uppercase background PSSMs for every acceptor, and the site counts."
    names = prefix + '_' + pd.Series(ACCEPTORS)
    bg = pd.DataFrame([get_bg_dict(df, a) for a in ACCEPTORS], index=names)
    bg_upper = pd.DataFrame([get_bg_dict(df, a, seq_col='site_seq_upper') for a in ACCEPTORS],
                            index=names + '_upper')
    cnt = {name: get_site_cnt(df, a) for a, name in zip(ACCEPTORS, names)}
    print(f'  {prefix} site counts:', cnt)
    return pd.concat([bg, bg_upper]), cnt


def main():
    print('== ks dataset background ==')
    ks = dedup_upper(kdata.load('ks_unique'))
    bg_ks, cnt_ks = build_background(ks, 'ks')

    print('\n== human phosphoproteome background ==')
    human = dedup_upper(kdata.load('human_site'))
    human['acceptor'] = human.site_seq.str[20].str.upper()
    print('  acceptors:', dict(human['acceptor'].value_counts()))
    bg_human, cnt_human = build_background(human, 'human')

    all_pssms = pd.concat([bg_ks, bg_human])
    print('\nbackground table:', all_pssms.shape)

    kdata.save('ks_background', all_pssms)

    # the CDDM log-odds (cddm_LO) divide CDDM by this background, but motif_01 is their single
    # producer (run it after this) — we don't refresh them here to keep one writer per dataset.

    print('\n== figures ==')
    # the first 5 rows of each block are the case-preserving S/T/Y/ST/STY backgrounds
    plot_logos(bg_ks.iloc[:5], cnt_ks, prefix='')
    save_pdf(FIG / 'background_ks.pdf')

    plot_logos(bg_human.iloc[:5], cnt_human, prefix='')
    save_pdf(FIG / 'background_human.pdf')

    plot_logo_heatmap(get_prob(human[human['acceptor'] == 'Y']),
                      title='Y sites in human phosphoproteome', figsize=(15, 7))
    save_svg(FIG / 'human_y_site.svg')
    print('figures written to', FIG)


if __name__ == '__main__':
    main()
