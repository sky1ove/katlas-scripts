"""kd_01b · Align every kinase domain, and label the catalytic residues.

Multiple-sequence-align all 5,536 kinase domains so that equivalent residues share a column
number. That pseudo-position index is what makes the domains comparable: a one-hot feature at
"column 1525" means the same structural position in every kinase.

The alignment itself is Clustal Omega, which is an external binary, not a Python call:

    clustalo -i out/kinase_domains.fasta -o out/kinase_domains.aln --force --outfmt=clu

The script writes the FASTA, runs clustalo if `RUN_CLUSTALO` is on (about an hour on 5.5k
sequences), and otherwise expects the `.aln` to already be there.

**Catalytic-residue labels.** Four alignment columns carry the canonical active-site motifs,
located by looking at which positions are most conserved:

    1525  D of the HRD motif         -> HRD_D1
    1549  N of the catalytic loop    -> catloop_N   (the N of HRD..K..N)
    1724  D of the DFG motif         -> DFG_D2
    2618  D around the D[IV]WS motif -> DWS_D3

A domain carrying both catalytic aspartates (HRD-D1 and DFG-D2) is treated as catalytically active
(`active_D1_D2`); that flag is what kd_02 filters on, and it is the difference between the 5,536
domains here and the 4,209 that carry features.

`validate_against_pseudo` checks the rule against the curated pseudokinase annotation in
kinase_info (the ground truth for whether a domain can catalyse): `active_D1_D2` flags 82% of the
annotated pseudokinases while keeping 94% of the active kinases. Adding the catalytic-loop Asn buys
nothing, and the VAIK β3 lysine (column 646) aligns too poorly in the N-lobe to help — it is present
in only ~65% of active kinases — so neither enters the rule.

Labels are joined on `kd_ID`, not by row position — `aln2df` indexes by sequence ID, so the two
tables cannot silently drift out of order.

Inputs   out/kd_uniprot.parquet (kd_01a); out/kinase_domains.aln (this script's own clustalo output)
Outputs  out/kinase_domains.fasta, out/kinase_domains.aln, out/kd_align.parquet,
         out/kd_motif_labeled.parquet, out/kd_align_freq_max_aa.csv

Run:  python nbs/kd_01b_alignment.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from paths import OUT

import kdata
from katlas.utils import aln2df, get_aln_freq, get_fasta, run_clustalo

FASTA = OUT / 'kinase_domains.fasta'
ALN = OUT / 'kinase_domains.aln'
RUN_CLUSTALO = False    # the alignment takes ~1h; the .aln is normally already present

#: alignment column -> catalytic-residue label and the expected residue
MOTIF_COLS = {'HRD_D1': (1525, 'D'),      # D of the HRD motif
              'catloop_N': (1549, 'N'),   # catalytic-loop Asn (the N of HRD..K..N)
              'DFG_D2': (1724, 'D'),      # D of the DFG motif
              'DWS_D3': (2618, 'D')}      # D around the D[IV]WS motif
ACTIVE_COLS = ['HRD_D1', 'DFG_D2']        # both catalytic aspartates required to count as active
ACTIVE_FLAG = 'active_D1_D2'              # name of the "both aspartates present" flag
VAIK_K_COL = 646                          # β3 lysine of the VAIK motif; reported by the pseudo check, not a filter

FREQ_CUT = 0.1      # report positions where one residue exceeds this frequency


def build_alignment():
    "Write the FASTA, optionally align, and read the alignment into a kd_ID x position frame."
    kd = pd.read_parquet(OUT / 'kd_uniprot.parquet')
    print('kinase domains:', kd.shape)

    get_fasta(kd, seq_col='kd_seq', id_col='kd_ID', path=FASTA)
    print('wrote', FASTA)

    if RUN_CLUSTALO:
        print('running clustalo (this takes about an hour)...')
        run_clustalo(input_fasta=FASTA, output_aln=ALN)
    if not ALN.exists():
        sys.exit(f'{ALN} not found - set RUN_CLUSTALO = True, or run clustalo by hand:\n'
                 f'  clustalo -i {FASTA} -o {ALN} --force --outfmt=clu')

    align = aln2df(ALN)
    print('alignment:', align.shape, f'({align.shape[1]} pseudo-positions)')
    return kd, align


def position_frequencies(align):
    "Most frequent residue at each alignment column, ignoring gaps."
    freq = get_aln_freq(align)
    freq = freq.iloc[1:, :]                       # drop the '-' (gap) row
    out = pd.concat([freq.idxmax(), freq.max()], axis=1)
    out.columns = ['aa', 'max_value']
    out = out.sort_values('max_value', ascending=False).reset_index(names='position')
    conserved = out[out.max_value > FREQ_CUT]
    print(f'positions with a residue above {FREQ_CUT:.0%}: {len(conserved)}')
    print(conserved.head(8).to_string(index=False))
    return out, conserved


def label_motifs(kd, align):
    "Flag the catalytic residues, joined on kd_ID so the two tables cannot drift out of order."
    labels = pd.DataFrame(index=align.index)
    for name, (pos, residue) in MOTIF_COLS.items():
        if pos not in align.columns:
            sys.exit(f'alignment has no column {pos} - the .aln does not match MOTIF_COLS')
        labels[name] = (align[pos] == residue).astype(int)

    kd = kd.set_index('kd_ID')
    missing = len(kd.index.difference(labels.index))
    if missing:
        print(f'  WARNING: {missing} domains are absent from the alignment')
    kd = kd.join(labels)

    kd[ACTIVE_FLAG] = (kd[ACTIVE_COLS].sum(axis=1) == len(ACTIVE_COLS)).astype(int)
    for name, (pos, residue) in MOTIF_COLS.items():
        print(f'  {name} (col {pos}, {residue}): {int(kd[name].sum())} domains')
    print(f'  {ACTIVE_FLAG} ({" & ".join(ACTIVE_COLS)}): {int(kd[ACTIVE_FLAG].sum())} / {len(kd)}')
    return kd.reset_index()


def report(kd):
    print('\nactive domains by annotation:')
    print(kd[kd[ACTIVE_FLAG] == 1].kd_note.value_counts().head(5).to_string())
    print('\ninactive domains by annotation:')
    print(kd[kd[ACTIVE_FLAG] == 0].kd_note.value_counts().head(5).to_string())

    human = kd[kd.Organism == 'Homo sapiens (Human)']
    per_uniprot = human.groupby('Uniprot').agg({'kd_ID': lambda x: ','.join(x.unique()),
                                                ACTIVE_FLAG: 'sum'})
    print(f'\nhuman: {len(human)} domains over {len(per_uniprot)} proteins, '
          f'{int(human[ACTIVE_FLAG].sum())} active')


def validate_against_pseudo(align):
    """Check the active_D1_D2 rule against the curated pseudokinase annotation.

    kinase_info flags each human kinase pseudo (1) or active (0) — the ground truth for whether a
    domain can catalyse. Joined to the alignment by kd_ID, it tells us how well each motif residue,
    and the rules built from them, separate the two. active_D1_D2 (HRD-D1 and DFG-D2) is the chosen
    rule; the catalytic-loop Asn adds nothing over it, and the VAIK β3 lysine aligns too poorly in
    the N-lobe to be a reliable filter — both are shown here rather than used.
    """
    ki = kdata.load('kinase_info').drop_duplicates('kd_ID')
    ki = ki[ki['pseudo'].isin(['0', '1'])]
    lab = ki[ki['kd_ID'].isin(align.index)].set_index('kd_ID')['pseudo'].astype(int)
    A = align.loc[lab.index]
    act, pse = (lab == 0).to_numpy(), (lab == 1).to_numpy()
    n_act, n_pse = int(act.sum()), int(pse.sum())
    print(f'\npseudo-annotation check: {n_act} active + {n_pse} pseudo human kinases (kinase_info)')

    checks = {'HRD_D1': (1525, 'D'), 'catloop_N': (1549, 'N'),
              'DFG_D2': (1724, 'D'), 'VAIK_K': (VAIK_K_COL, 'K')}
    print('  residue     active  pseudo   all')
    for name, (col, aa) in checks.items():
        hit = (A[col] == aa).to_numpy()
        print(f'    {name:9} {hit[act].mean():6.0%}  {hit[pse].mean():6.0%}  {(align[col] == aa).mean():6.0%}')

    D1, N1 = (A[1525] == 'D').to_numpy(), (A[1549] == 'N').to_numpy()
    D2, K = (A[1724] == 'D').to_numpy(), (A[VAIK_K_COL] == 'K').to_numpy()

    def rule(name, keep):                     # `keep` = predicted active; a pseudokinase should fail it
        caught, kept = int((~keep & pse).sum()), int((keep & act).sum())
        acc = (kept + caught) / len(keep)
        print(f'    {name:24} pseudo caught {caught:2}/{n_pse}  active kept {kept:3}/{n_act}  acc {acc:.3f}')

    print('  rule:')
    rule('active_D1_D2 = HRD_D1 & DFG_D2', D1 & D2)
    rule('+ catloop_N', D1 & N1 & D2)
    rule('+ VAIK_K', D1 & N1 & D2 & K)

    ct = pd.crosstab(pd.Series(np.where(D1 & D2, 'active_D1_D2', 'fails'), name='rule'),
                     pd.Series(np.where(pse, 'pseudo', 'active'), name='annotation'))
    print('  active_D1_D2 x annotation:')
    print(ct.to_string())


def main():
    OUT.mkdir(exist_ok=True)

    kd, align = build_alignment()

    align.columns = align.columns.astype(str)     # parquet needs string column names
    align.to_parquet(OUT / 'kd_align.parquet')
    print('wrote', OUT / 'kd_align.parquet')
    align.columns = align.columns.astype(int)

    _, conserved = position_frequencies(align)
    conserved.to_csv(OUT / 'kd_align_freq_max_aa.csv', index=False)

    kd = label_motifs(kd, align)
    kd.to_parquet(OUT / 'kd_motif_labeled.parquet', index=False)
    print('\nwrote', OUT / 'kd_motif_labeled.parquet', kd.shape)
    report(kd)
    validate_against_pseudo(align)


if __name__ == '__main__':
    main()
