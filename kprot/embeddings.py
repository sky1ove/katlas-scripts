"""Protein embedding extraction helpers for ESM2 and ProtT5 backbones."""


import gc
import re

import pandas as pd
import torch
from tqdm.notebook import tqdm

tqdm.pandas()

def_device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def get_esm(
    df: pd.DataFrame,  # dataframe containing protein sequences
    col: str,  # column containing amino acid sequences
    model_name: str = "esm2_t33_650M_UR50D",
    batch_size: int = 1,  # number of sequences per inference batch
) -> pd.DataFrame:
    "Extract ESM2 embeddings (mean pooled per sequence)."
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
    model = model.to(def_device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()
    match = re.search(r"_t(\d+)_", model_name)
    if not match:
        raise ValueError(f"Cannot infer repr layer from {model_name}")
    layer = int(match.group(1))

    print(f"Using ESM layer {layer}")
    print("Available models\n"
          "esm2_t48_15B_UR50D\n"
          "esm2_t36_3B_UR50D\n"
          "esm2_t33_650M_UR50D\n"
          "esm2_t30_150M_UR50D\n"
          "esm2_t12_35M_UR50D\n"
          "esm2_t6_8M_UR50D\n")

    sequences = df[col].tolist()
    all_embeddings: list[object] = []

    for start in tqdm(range(0, len(sequences), batch_size)):
        batch_seqs = sequences[start : start + batch_size]
        data = [(f"seq_{idx}", seq) for idx, seq in enumerate(batch_seqs)]

        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(def_device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[layer], return_contacts=False)

        token_reps = results["representations"][layer]
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        for row_idx, seq_len in enumerate(batch_lens):
            emb = token_reps[row_idx, 1 : seq_len - 1].mean(0)
            all_embeddings.append(emb.cpu().numpy())

        del results, token_reps, batch_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return pd.DataFrame(
        all_embeddings,
        index=df.index,
        columns=[f"esm_{i}" for i in range(len(all_embeddings[0]))],
    )

def get_t5(
    df: pd.DataFrame,  # dataframe containing protein sequences
    col: str = "sequence",  # column containing amino acid sequences
) -> pd.DataFrame:
    "Extract ProtT5-XL-uniref50 embeddings from protein sequences in a dataframe."
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(def_device)
    model.half()

    def t5_embeddings(sequence: str) -> object:
        seq_len = len(sequence)
        sequence_tokens = [" ".join(list(re.sub(r"[UZOB]", "X", sequence)))]
        ids = tokenizer.batch_encode_plus(sequence_tokens, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids["input_ids"]).to(def_device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(def_device)

        with torch.no_grad():
            embedding_rpr = model(input_ids=input_ids, attention_mask=attention_mask)

        return embedding_rpr.last_hidden_state[0][:seq_len].detach().to(torch.float32).cpu().numpy().mean(axis=0)

    series = df[col].progress_apply(t5_embeddings)
    t5_feature = pd.DataFrame(series.tolist(), index=df.index)
    t5_feature.columns = "T5_" + t5_feature.columns.astype(str)
    return t5_feature

