"""kd_07 · Predict substrate motifs for kinase domains that have none — by retrieval.

The shipped model (`SHIPPED_MODEL`, default k-NN retrieval on the one-hot alignment features) predicts
each unlabelled active domain's motif as the inverse-distance-weighted mean of its nearest labelled
kinases' PSSMs, over the **whole labelled set** (dev + held-out test — the split was only for model
selection). The paper's target is domains *very similar* to characterised kinases — exactly where
retrieval is most natural and interpretable ("this domain is nearly identical to kinase X, so its motif
≈ X's"). A tree ensemble edges retrieval only on domains *far* from any labelled kinase (kd_05d), the
low-confidence regime we do not rely on. The distance to the nearest labelled kinase rides along as the
per-domain confidence, so the near (high-confidence) predictions are trusted and far ones treated with
caution (kd_06a shows how reliability falls with that distance).

`SHIPPED_MODEL` is 'kNN' or any model `kd_util.build_estimator` supports (RandomForest / ExtraTrees /
Ridge / …), refit on the whole labelled set — swap it in one place, or with `--model`, to redeploy.
Runs for both shipped targets — the PSPA peptide-array motif and the CDDM MLP-attribution motif; kd_08
UMAPs both next to the measured set.

Inputs   out/kd_train_{pspa,mlp_attr}_onehot.parquet (kd_03), out/kd_feat_onehot.parquet (kd_02a),
         out/kd_model_selection.parquet (kd_04b, for the tuned config)
Outputs  out/kd_pred_new_{pspa,mlp_attr}.parquet

Run:  python nbs/kd_07_predict_novel.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kd_util
import pandas as pd
from paths import OUT

import kdata

TARGETS = ['pspa', 'mlp_attr', 'cddm']
SHIPPED_MODEL = 'kNN'       # deployment predictor: 'kNN' retrieval, or any kd_util.build_estimator model


def run(label, feat_all):
    print(f'\n== {label.upper()} ({SHIPPED_MODEL}) ==')
    df, feat_col, target_col = kd_util.load_train(label, 'onehot')
    ref_ids = df.iloc[:, 0].to_numpy()                     # ALL labelled kinases (dev + test) — refit on all
    X_ref, Y_ref = df[feat_col].to_numpy(float), df[target_col].to_numpy(float)

    query = feat_all.index.difference(ref_ids)             # active domains with no measured PSSM
    pred, nnd = kd_util.deploy_predict(SHIPPED_MODEL, X_ref, Y_ref,
                                       feat_all.loc[query, feat_col].to_numpy(float), label)

    out = pd.DataFrame(pred, index=query, columns=target_col)
    out['nn_dist'] = nnd
    org = kdata.load('kd_uniprot').set_index('kd_ID')['Organism']       # keep species (dark human + non-human)
    out['organism'] = org.reindex(out.index)
    out['is_human'] = out.organism.str.contains('Homo sapiens', na=False)
    path = OUT / f'kd_pred_new_{label}.parquet'
    out.to_parquet(path)
    print(f'  predicted {len(query):,} unlabelled active domains -> {path}')
    print(f'  nn_dist: median {out.nn_dist.median():.2f} '
          f'(near-half ≤ {out.nn_dist.median():.2f}, far-half above)')


def main():
    import argparse
    global SHIPPED_MODEL
    p = argparse.ArgumentParser(description='Deploy the shipped predictor over all unlabelled domains')
    p.add_argument('--model', default=SHIPPED_MODEL, help='deployment model (build_estimator name, or kNN)')
    SHIPPED_MODEL = p.parse_args().model
    feat_all = pd.read_parquet(OUT / 'kd_feat_onehot.parquet')     # one-hot for all active domains
    for label in TARGETS:
        run(label, feat_all)


if __name__ == '__main__':
    main()
