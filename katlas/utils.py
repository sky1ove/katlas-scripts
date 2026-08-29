"""Functions to preprocess sequence to prepare kinase substrate dataset"""


import numpy as np, pandas as pd
from tqdm import tqdm
from .data import Data
from pathlib import Path
from kplot.utils import get_color_dict, save_show, save_svg
from sklearn.preprocessing import StandardScaler

# for alignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO, AlignIO
import subprocess


sty_color=get_color_dict(['S','T','Y'])| get_color_dict(['s','t','y'])

group_color=get_color_dict(
            ['CMGC','AGC', # blue
             'TK','TKL', # orange
             'CAMK','STE', # green
             'CK1', 'NEK', # red
             'Atypical','Other', # purple
             'RGC'
            ]
)

def get_subfamily_color():
    group_color2 = pd.DataFrame(group_color).T
    group_color2 = group_color2.reset_index(names='group')
    info=Data.kinase_info()
    subfamily_color = info[['group','subfamily']].merge(group_color2).drop(columns=['group']).set_index('subfamily')
    subfamily_color = subfamily_color.apply(tuple, axis=1).to_dict()
    return subfamily_color

pspa_category_color = get_color_dict(['Basophilic', 'Pro-directed', 'Acidophilic', 'Map3k', 'Map4k',
       'Alpha/mlk', 'Fgf and vegf receptors', 'Assorted', 'Ripk/wnk', 'Pkc',
       'Ephrin receptors', 'Eif2ak/tlk', 'Nek/ask', 'Pdgf receptors', 'Src',
       'Jak', 'Ulk/ttbk', 'Cmgc', 'Tec', 'Tam receptors'])


def STY2sty(input_string: str):
    "Replace all uppercase S/T/Y with lowercase s/t/y in a sequence."
    return input_string.replace("S", "s").replace("T", "t").replace("Y", "y")

def pSTY2sty(string):
    "Convert pS/pT/pY to s/t/y in a string."
    return string.replace("pS", "s").replace("pT", "t").replace("pY", "y")

def sty2pSTY(string):
    "Convert s/t/y to pS/pT/pY in a string."
    return string.replace("s", "pS").replace("t", "pT").replace("y", "pY")

def sty2pSTY_df(df):
    "Apply sty to pSTY conversion to a dataframe index."
    df = df.copy()
    df.index = df.index.map(sty2pSTY)
    return df

def check_seq(seq):
    """Convert non-s/t/y characters to uppercase and replace disallowed characters with underscores."""
    acceptor = seq[len(seq) // 2]
    if acceptor.lower() not in {"s","t","y"}: raise ValueError(f"Center must be s/t/y; got {acceptor} in {seq!r}")

    allowed_chars = set("PGACSTVILMFYWHKRQNDEsty")
    return "".join(char if char in {'s', 't', 'y'} else (char.upper() if char.upper() in allowed_chars else '_') for char in seq)

def check_seqs(data,col=None):
    "Convert non-s/t/y to upper case & replace with underscore if the character is not in the allowed set"
    if isinstance(data, pd.DataFrame):
        if col is None: raise ValueError("Must specify 'col' when passing a DataFrame.")
        seqs = data[col]
    elif isinstance(data, (pd.Series, list)): seqs = pd.Series(data)
    else: raise TypeError("Input must be a DataFrame, Series, or list.")
    
    lengths = seqs.str.len().value_counts()
    if len(lengths) != 1: raise ValueError(f"Inconsistent sequence lengths detected: {lengths.to_dict()}")
    return seqs.apply(check_seq)

def validate_site(site_info,
                  seq):
    "Validate site position residue match with site residue."
    pos=int(site_info[1:])-1 # python index starts from zero
    if pos >= len(seq) or pos < 0: 
        return int(False)
    return int(seq[pos]==site_info[0])

def validate_site_df(df, 
                     site_info_col,
                     protein_seq_col): 
    "Validate site position residue match with site residue in a dataframe."
    return df.apply(lambda r: validate_site(r[site_info_col],r[protein_seq_col]) , axis=1)

def phosphorylate_seq(seq, # full protein sequence
                      *sites, # site info, e.g., S140
                      ):
    "Phosphorylate protein sequence based on phosphosites (e.g.,S140). "
    seq = list(seq)

    for site in sites:
        char = site[0] 
        position = int(site[1:]) - 1 # substract 1 as python index starts from 0

        if 0 <= position < len(seq):
            if seq[position] == char:
                seq[position] = char.lower()  
            else:
                raise ValueError(f"Mismatch at position {position+1}: expected {char}, found {seq[position]}")
        else:
            raise IndexError(f"Position {position+1} out of range for sequence length {len(seq)}")

    return ''.join(seq)

def phosphorylate_seq_df(df,
                         id_col='substrate_uniprot', # column of sequence ID
                         seq_col='substrate_sequence', # column that contains protein sequence
                         site_col='site', # column that contains site info, e.g., S140
                         
                        ):
    "Phosphorylate whole sequence based on phosphosites in a dataframe"
    df_seq = df.groupby(id_col).agg({site_col:lambda r: r.unique(),seq_col:'first'}).reset_index()
    df_seq['phosphoseq'] = df_seq.apply(lambda r: phosphorylate_seq(r[seq_col],*r[site_col]),axis=1)
    return df_seq

def extract_site_seq(df: pd.DataFrame, # dataframe that contains protein sequence
                     seq_col: str, # column name of protein sequence
                     site_col: str, # column name of site information (e.g., S10)
                     n=7, # length of surrounding sequence (default -7 to +7)
                    ):
    "Extract -n to +n site sequence from protein sequence"
    
    data = []
    for i, r in tqdm(df.iterrows(),total=len(df)):
        position = int(r[site_col][1:]) - 1
        start = position - n
        end = position + n +1

        # Extract the subsequence
        subseq = r[seq_col][max(0, start):min(len(r[seq_col]), end)]

        # Pad the subsequence if needed
        if start < 0:
            subseq = "_" * abs(start) + subseq
        if end > len(r[seq_col]):
            subseq = subseq + "_" * (end - len(r[seq_col]))

        data.append(subseq)
        
    return np.array(data)

def get_fasta(df,seq_col='kd_seq',id_col='kd_ID',path='out.fasta'):
    "Generate fasta file from sequences."
    records = [
        SeqRecord(Seq(str(row[seq_col])), id=str(row[id_col]), description="")
        for _, row in df.iterrows()
    ]
    SeqIO.write(records, path, "fasta")
    print(len(records))

def run_clustalo(input_fasta,  # .fasta fname
                 output_aln, # .aln output fname
                 outfmt="clu"):
    "Run Clustal Omega to perform multiple sequence alignment."
    output_aln = Path(output_aln)
    output_aln.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "clustalo", "-i", str(input_fasta),
        "-o", str(output_aln),
        "--force", f"--outfmt={outfmt}"
    ], check=True)

def aln2df(fname):
    alignment = AlignIO.read(fname, "clustal")
    alignment_array = [list(str(record.seq)) for record in alignment]
    ids = [record.id for record in alignment]
    df = pd.DataFrame(alignment_array, index=ids)
    df.columns = df.columns+1
    return df

def get_aln_freq(df):
    "Get frequency of each amino acid across each position from the aln2df output."
    counts_df = df.apply(lambda col: col.value_counts(), axis=0).fillna(0)
    return counts_df.div(counts_df.sum(axis=0), axis=1)
