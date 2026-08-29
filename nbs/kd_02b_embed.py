"""kd_02b · Protein-language-model embeddings for each active kinase domain.

ProtT5 and ESM-2 embeddings of the domain sequence — one ~1k-dimensional vector per catalytically
active domain (`active_D1_D2`), the learned counterpart to the one-hot alignment features (kd_02a)
that kd_04b puts head to head.

The embeddings are model inference over the 4,209 sequences, so they are read from
`raw/{t5,esm}_kd.parquet` when present (they are) and only recomputed with `--embed`; ESM in
particular needs the `esm` package, which is not currently installed.

Inputs   out/kd_motif_labeled.parquet (kd_01b), raw/t5_kd.parquet, raw/esm_kd.parquet
Outputs  out/kd_feat_t5.parquet, out/kd_feat_esm.parquet

Run:  python nbs/kd_02b_embed.py [--embed]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from paths import OUT, RAW


def active_domains():
    "The active (active_D1_D2) kd rows, carrying kd_seq for the embedding models."
    kd = pd.read_parquet(OUT / 'kd_motif_labeled.parquet')
    kd = kd[kd.active_D1_D2.astype(bool)].reset_index(drop=True)
    print('active domains:', len(kd))
    return kd


def embeddings(kd_active, recompute):
    "ProtT5 and ESM-2 embeddings of the domain sequences (cached in raw/ by default)."
    out = {}
    for name, cache in [('t5', RAW / 't5_kd.parquet'), ('esm', RAW / 'esm_kd.parquet')]:
        if not recompute and cache.exists():
            feat = pd.read_parquet(cache)
            print(f'  {name}: {feat.shape} (cached, {cache})')
        else:
            from kprot.embeddings import get_esm, get_t5
            print(f'  {name}: computing over {len(kd_active)} domains...')
            feat = (get_t5 if name == 't5' else get_esm)(kd_active, 'kd_seq')
            feat.index = kd_active.kd_ID
            feat.to_parquet(cache)
        out[name] = feat
    return out


def main():
    recompute = '--embed' in sys.argv
    kd_active = active_domains()

    print('embeddings:')
    for name, feat in embeddings(kd_active, recompute).items():
        p = OUT / f'kd_feat_{name}.parquet'
        feat.to_parquet(p)
        print(f'  wrote {p} {feat.shape}')


if __name__ == '__main__':
    main()
