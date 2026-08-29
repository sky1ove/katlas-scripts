"""Load various kinase-relatd datasets"""


import os
import re
import pandas as pd
from filelock import FileLock
from functools import lru_cache
from fastcore.all import patch

import gdown,zipfile,shutil,tempfile
from pathlib import Path

def _normalize_dir(path: str | Path) -> Path:
    "Get absolute path to the dataset directory, and ensure it ends with a katlas_dataset(s) folder."

    # expanduser can expand ~, and resolve can return the absolute path
    resolved_path = Path(path).expanduser().resolve()
    # accept both the canonical "katlas_dataset" and the repo's "katlas_datasets" folder
    if resolved_path.name in ("katlas_dataset", "katlas_datasets"): return resolved_path
    return resolved_path / "katlas_dataset"

def _dataset_dir() -> Path:
    "Dataset directory: the KATLAS_DATA_DIR env var, else the repo's katlas_datasets/ (next to the package), else tmp."

    # get the environment path if stored in KATLAS_DATA_DIR
    env_path = os.getenv("KATLAS_DATA_DIR")

    # if env path exists, get the absolute path of it with /katlas_dataset at the end
    if env_path: return _normalize_dir(env_path)

    # dev-repo fallback: katlas_datasets/ inside the repo. This avoids the surprising tmp directory when
    # scripts are run without KATLAS_DATA_DIR set (the repo's .env is not always loaded). The store lives
    # under nbs/ so the analysis scripts can read it as a plain folder; the old top-level spot still works.
    repo = Path(__file__).resolve().parents[1]
    for repo_data in (repo / "nbs" / "katlas_datasets", repo / "katlas_datasets"):
        if repo_data.exists(): return repo_data

    # last resort: tmp directory with /katlas_dataset at the end
    return Path(tempfile.gettempdir()) / "katlas_dataset"


class Data:
    "A class for fetching various datasets."
    DATASET_DIR = _dataset_dir()

def _normalize_required_fname(
    required_files: str | Path | list[str | Path] | None,  # Required dataset members
) -> tuple[str, ...]:
    "If required files are specified, return a tuple of file names."
    # if nothing is required, return an empty tuple
    if required_files is None: return tuple()

    # if single file, put it in a list; if list, keep it then convert to file name strings.
    items = [required_files] if isinstance(required_files, (str, Path)) else required_files
    return tuple(str(Path(item)) for item in items)

@lru_cache
def _read_file(path: str) -> pd.DataFrame:
    "Read a dataset file with caching."

    file_path = Path(path)
    # suffix returns extension only (e.g., .csv, .parquet)
    ext = file_path.suffix.lower() # case to lower to ensure file extension is case insensitive

    if ext == ".csv": df = pd.read_csv(file_path)
    elif ext == ".parquet": df = pd.read_parquet(file_path)
    else: raise ValueError(f"Unsupported file type: {ext}. Supported types: .csv, .parquet")

    # return dataframe, TODO: change the source data, make sure no unnamed column
    return df.rename(columns={"Unnamed: 0": "kinase"}) if "Unnamed: 0" in df.columns else df

@patch(cls_method=True)
def clear_cache(cls: Data) -> None:
    "Clear the cache of the _read_file."
    _read_file.cache_clear()

@patch(cls_method=True)
def download(
    cls: Data,  # Patched class receiver
    download_dir: str | Path | None = None,  # Parent directory or katlas_dataset folder
    force: bool = False,  # Re-download even if the folder already exists
    verbose: bool = True,  # Print status messages
    required_files: str | Path | list[str | Path] | None = None,  # Files that must exist after download
) -> None:
    """Download dataset zip and extract to folder."""

    url = "https://drive.google.com/uc?id=17wIl0DbdoHV036Z3xgaT_0H3LlM_W47l"

    # if download_dir is provided, update cls DATASET_DIR to it, otherwise use tmp as default
    if download_dir is not None: cls.DATASET_DIR = _normalize_dir(download_dir)

    dataset_dir = cls.DATASET_DIR
    zip_path = dataset_dir.parent / "katlas_dataset.zip"

    # set a lock path for FileLock, so that if multiple processes try to download, the one that come first will create the lock and others have to wait til the block is done.
    lock_path = dataset_dir.parent / "katlas_dataset.lock"

    # get required file list to check missing files
    required_list = _normalize_required_fname(required_files)

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        missing_files = [rel_path for rel_path in required_list if not (dataset_dir / rel_path).exists()]

        # if force is True, or dataset folder exist, or no missing file, no need to download
        needs_download = force or not dataset_dir.exists() or bool(missing_files)

        if not needs_download:
            if verbose: print(f"✅ Dataset exists at: {dataset_dir}")
            return

        # prepare to download, clear the cache first
        cls.clear_cache()

        if dataset_dir.exists(): # if dataset folder exists
            if verbose:
                if force: print(f"♻️ Removing existing folder: {dataset_dir}")
                else: print(f"♻️ Dataset is missing {missing_files}; re-downloading to {dataset_dir}")
            # remove existing dataset folder
            shutil.rmtree(dataset_dir)

        # create empty dataset folder
        dataset_dir.mkdir(parents=True, exist_ok=True)

        if verbose: print("⬇️ Downloading katlas_dataset.zip ...")

        # download from Google Drive using gdown, save to zip_path
        downloaded_file = gdown.download(url, output=str(zip_path), quiet=not verbose)

        if downloaded_file is None or not Path(downloaded_file).exists():
            raise RuntimeError(
                "Dataset download failed. "
                "Please check your internet connection or Google Drive permissions."
            )

        if verbose: print(f"📂 Extracting to {dataset_dir} ...")

        with zipfile.ZipFile(downloaded_file, "r") as zip_ref:

            # safety check, prevent files write outside of the dataset folder
            for member in zip_ref.namelist():
                member_path = dataset_dir / member
                # resolve() can solve relative path to absolution path (../.. to upper level)
                # is_relative_to() can check if the member_path is within the dataset folder
                if not member_path.resolve().is_relative_to(dataset_dir.resolve()):
                    # if not within dataset folder, it means zip files try to write outside of the folder, so stop
                    raise RuntimeError(f"Unsafe zip file detected (zip-slip): {member}")
            # extract all files to the dataset folder
            zip_ref.extractall(dataset_dir)

        wrapped_dir = dataset_dir / "katlas_datasets"
        if wrapped_dir.exists():
            for item in wrapped_dir.iterdir():
                shutil.move(str(item), str(dataset_dir / item.name))
            shutil.rmtree(wrapped_dir)

        macosx_dir = dataset_dir / "__MACOSX"
        if macosx_dir.exists():
            shutil.rmtree(macosx_dir)

        # check if required files are in the extracted dataset, if not, raise error
        missing_after = [rel_path for rel_path in required_list if not (dataset_dir / rel_path).exists()]
        if missing_after:
            raise FileNotFoundError(f"Dataset download completed, but these files are still missing: {missing_after}")

        try:
            if verbose: print(f"🧹 Removing zip file: {downloaded_file}")
            # remove the downloaded zip file to save space
            Path(downloaded_file).unlink()

        except Exception as e:
            if verbose: print(f"⚠️ Could not remove {downloaded_file}: {e}")

        if verbose: print(f"✅ Done! Extracted dataset is at: {dataset_dir}")

@patch(cls_method=True)
def read_file(
    cls: Data,  # Patched class receiver
    rel_path: str | Path,  # Relative path inside katlas_dataset
    auto_download: bool = True,  # Download the dataset bundle if the file is missing
) -> pd.DataFrame:
    "Load a CSV or Parquet file from the local dataset folder; save cache when reading and return a copy of the cached data."

    rel_path_str = str(Path(rel_path))
    if auto_download: cls.download(verbose=False, required_files=rel_path_str)

    path = cls.DATASET_DIR / rel_path_str
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. "
            f"Call Data.download(required_files={rel_path_str!r}) or pass auto_download=True."
        )

    # read file from the cache, so even if change the file, it won't change the cached data
    return _read_file(str(path.resolve())).copy()

_DATASET_PATHS = {
    "kinase_info": "kinase_info.csv",
    "kinase_uniprot": "uniprot_human_keyword_kinase.parquet",
    "kd_uniprot": "uniprot_kd_labeled.parquet",
    "pspa_tyr": "PSPA/pspa_tyr_norm.parquet",
    "pspa_st": "PSPA/pspa_st_norm.parquet",
    "pspa": "PSPA/pspa_all_norm.parquet",
    "pspa_raw": "PSPA/pspa_all_raw.parquet",
    "pspa_scale": "PSPA/pspa_all_scale.parquet",
    "pspa_st_pct": "PSPA/pspa_pct_st.parquet",
    "pspa_tyr_pct": "PSPA/pspa_pct_tyr.parquet",
    "num_dict": "PSPA/pspa_divide_num.csv",
    "ks_unique": "CDDM/unique_ks_sites.parquet",
    "ks_dataset": "CDDM/ks_datasets.parquet",
    "ks_background": "CDDM/ks_background.parquet",
    "cddm": "CDDM/pssms.parquet",
    "cddm_upper": "CDDM/pssms_upper.parquet",
    "cddm_LO": "CDDM/pssms_LO.parquet",
    "cddm_LO_upper": "CDDM/pssms_LO_upper.parquet",
    "aa_info": "amino_acids/aa_info.parquet",
    "aa_rdkit": "amino_acids/aa_rdkit.parquet",
    "aa_morgan": "amino_acids/aa_morgan.parquet",
    "cptac_ensembl_site": "phosphosites/linkedOmicsKB_ref_pan.parquet",
    "cptac_unique_site": "phosphosites/cptac_unique_site.parquet",
    "cptac_gene_site": "phosphosites/linkedOmics_ref_pan.parquet",
    "psp_human_site": "phosphosites/psp_human.parquet",
    "ochoa_site": "phosphosites/ochoa_site.parquet",
    "combine_site_psp_ochoa": "phosphosites/combine_site_psp_ochoa.parquet",
    "combine_site_phosphorylated": "phosphosites/phosphorylated_combine_site.parquet",
    "human_site": "phosphosites/phosphorylated_combine_site20.parquet",
    "reactome_pathway_lo": "reactome_lowest_level.parquet",
    "reactome_pathway": "reactome_all_levels.parquet",
}

# Locally regenerated artifacts should take precedence over the packaged
# downloadable dataset bundle used by the public API.
_LOCAL_DATASET_PATHS = {
    "ks_unique": "nbs/raw/processed/pssm/unique_ks_sites.parquet",
    "ks_dataset": "nbs/raw/processed/ks_datasets_20260517.parquet",
    "ks_background": "nbs/raw/processed/pssm/ks_background.parquet",
    "cddm": "nbs/raw/cddm/CDDM_pssms.parquet",
    "cddm_upper": "nbs/raw/cddm/CDDM_pssms_upper.parquet",
    "cddm_LO": "nbs/raw/cddm/CDDM_pssms_LO.parquet",
    "cddm_LO_upper": "nbs/raw/cddm/CDDM_pssms_LO_upper.parquet",
    "human_site": "nbs/raw/processed/phosphosites/human_phosphoproteome.parquet",
}

# Datasets that are re-generated over time and saved with a date suffix
# (e.g. CDDM/pssms_20260618.parquet, CDDM/ks_datasets_20260517.parquet). For
# these the newest dated file is preferred; when none is present the registered
# _DATASET_PATHS entry is the fallback.
_VERSIONED_DATASETS = {
    # dataset_name: (directory relative to DATASET_DIR, file stem, extension)
    "kinase_info": ("", "kinase_info", ".csv"),
    "ks_dataset": ("CDDM", "ks_datasets", ".parquet"),
    "ks_background": ("CDDM", "ks_background", ".parquet"),
    "cddm": ("CDDM", "pssms", ".parquet"),
    "cddm_upper": ("CDDM", "pssms_upper", ".parquet"),
    "cddm_LO": ("CDDM", "pssms_LO", ".parquet"),
    "cddm_LO_upper": ("CDDM", "pssms_LO_upper", ".parquet"),
}


def _latest_versioned_rel(
    cls: type[Data],  # Data-like class providing DATASET_DIR
    dir_rel: str,  # Directory relative to DATASET_DIR ("" for the root, e.g. "CDDM")
    stem: str,  # File stem (e.g. "pssms")
    fallback_rel: str,  # Path to use when no dated file is found
    ext: str = ".parquet",  # File extension
) -> str:
    "Return the relative path of the newest `{stem}_YYYYMMDD{ext}` in `dir_rel`; fall back to `fallback_rel`."
    dir_path = cls.DATASET_DIR / dir_rel if dir_rel else cls.DATASET_DIR
    if not dir_path.exists():
        return fallback_rel

    # only an exactly-8-digit date suffix counts, so e.g. stem "pssms" never
    # picks up "pssms_upper"/"pssms_LO", and "ks_datasets" skips the
    # "ks_datasets_20250407_web" variant.
    dated_re = re.compile(rf"^{re.escape(stem)}_(\d{{8}})$")
    dated = [(m.group(1), p.name)
             for p in dir_path.glob(f"{stem}_*{ext}")
             if (m := dated_re.match(p.stem))]
    if dated:
        name = max(dated)[1]
        return f"{dir_rel}/{name}" if dir_rel else name
    return fallback_rel


def _read_named_data(
    cls: type[Data],  # Data-like class providing read_file
    dataset_name: str,  # Registry key from _DATASET_PATHS
) -> pd.DataFrame:
    "Load a dataset based on _DATASET_PATHS."
    local_path = _LOCAL_DATASET_PATHS.get(dataset_name)
    if local_path is not None:
        repo_path = Path(__file__).resolve().parents[1] / local_path
        if repo_path.exists():
            return _read_file(str(repo_path.resolve())).copy()

    try: rel_path = _DATASET_PATHS[dataset_name]
    except KeyError as exc: raise KeyError(f"Unknown dataset name: {dataset_name}") from exc

    if dataset_name in _VERSIONED_DATASETS:
        dir_rel, stem, ext = _VERSIONED_DATASETS[dataset_name]
        rel_path = _latest_versioned_rel(cls, dir_rel, stem, rel_path, ext)

    return cls.read_file(rel_path)

@patch(cls_method=True)
def kinase_info(cls: Data) -> pd.DataFrame:
    """
    Get information of 523 human kinases on kinome tree.
    Group, family, and subfamily classifications are sourced from Coral;
    full protein sequences are retrieved using UniProt IDs;
    kinase domain sequences are obtained from KinaseDomain.com;
    and cellular localization data is extracted from published literature.

    Reads the newest dated kinase_info_YYYYMMDD.csv (see 04b) when present, else kinase_info.csv.
    The `group` column holds the curated grouping; `group_old` keeps the original Coral grouping.
    """
    return _read_named_data(cls, "kinase_info")

@patch(cls_method=True)
def kinase_uniprot(cls: Data) -> pd.DataFrame:
    """
    Get information of 672 uniprot human kinases, which were retrieved from UniProt by filtering all human protein entries using the keyword 'kinase'.
    It includes additional pseudokinases and lipid kinases.
    """
    return _read_named_data(cls, "kinase_uniprot")

@patch(cls_method=True)
def kd_uniprot(cls: Data) -> pd.DataFrame:
    "Kinase domains extracted from UniProt database. "
    return _read_named_data(cls, "kd_uniprot")

@patch(cls_method=True)
def pspa_tyr(cls: Data) -> pd.DataFrame:
    """Get PSPA normalized data of tyrosine kinase."""
    return _read_named_data(cls, "pspa_tyr")

@patch(cls_method=True)
def pspa_st(cls: Data) -> pd.DataFrame:
    """Get PSPA normalized data of serine/threonine kinase."""
    return _read_named_data(cls, "pspa_st")

@patch(cls_method=True)
def pspa(cls: Data) -> pd.DataFrame:
    """Get PSPA normalized data of serine/threonine and tyrosine kinases.
    pS is duplicate of pT;
    For all kinases:
        -pS is duplicate of pT
    S/T kinases:
        - Column normalized to 17 (20 standard-S,T,C) or 16 (20-S,T,C,Y;PDHK1 & PDHK4) amino acids.
        - C is median of 17 or 16 (PDHK1 & PDHK4) amino acids
    TK (including _TYR which is non-canonical):
        - Column normalized to 18 (20-Y,C; canonical +TNNI3K,WEE1) or 16 (20-S,T,Y,C; IRR, JAK3, MST1R + non-canonical _TYR) aa.
        - Y is duplicate of F;
        - C is median of 18 or 16 amino acids.
"""
    return _read_named_data(cls, "pspa")

@patch(cls_method=True)
def pspa_raw(cls: Data) -> pd.DataFrame:
    """Get PSPA raw data of serine/threonine and tyrosine kinases."""
    df = _read_named_data(cls, "pspa_raw")
    # drop s columns, as it's a duplicate t
    df = df.loc[:, ~df.columns.str.endswith("s")]
    return df.copy()

@patch(cls_method=True)
def pspa_scale(cls: Data) -> pd.DataFrame:
    """
    Get PSPA (-5 to +4) scaled data from PSPA normalized data.
    Each position (including both pS/pT and pS=pT) are normalized to 1.
    """
    return _read_named_data(cls, "pspa_scale")

@patch(cls_method=True)
def pspa_st_pct(cls: Data) -> pd.DataFrame:
    """Get PSPA reference score to calculate percentile for serine/threonine kinases."""
    return _read_named_data(cls, "pspa_st_pct")

@patch(cls_method=True)
def pspa_tyr_pct(cls: Data) -> pd.DataFrame:
    """Get PSPA reference score to calculate percentile for tyrosine kinases."""
    return _read_named_data(cls, "pspa_tyr_pct")

@patch(cls_method=True)
def num_dict(cls: Data) -> dict[str, int]:
    """Get a dictionary mapping kinase to number of random amino acids in PSPA."""
    return _read_named_data(cls, "num_dict").set_index("kinase")["num_random_aa"].to_dict()

@patch(cls_method=True)
def ks_unique(cls: Data) -> pd.DataFrame:
    """Get kinase substrate dataset with unique sub site ID."""
    return _read_named_data(cls, "ks_unique")

@patch(cls_method=True)
def ks_dataset(cls: Data, thr: int | None = 40) -> pd.DataFrame:
    "Get kinase-substrate pairs collected from public resources (newest dated CDDM/ks_datasets_YYYYMMDD.parquet)."
    df = _read_named_data(cls, "ks_dataset")
    if thr is None: return df
    return df[df["num_kin"] <= thr].copy()

@patch(cls_method=True)
def ks_background(cls: Data) -> pd.DataFrame:
    """Get kinase substrate dataset with unique sub site ID."""
    return _read_named_data(cls, "ks_background")

@patch(cls_method=True)
def cddm(cls: Data) -> pd.DataFrame:
    """Get the CDDM dataset (newest dated CDDM/pssms_YYYYMMDD.parquet, else CDDM/pssms.parquet)."""
    return _read_named_data(cls, "cddm")

@patch(cls_method=True)
def cddm_upper(cls: Data) -> pd.DataFrame:
    """Get the all-uppercase-sequence CDDM dataset (newest dated CDDM/pssms_upper_YYYYMMDD.parquet, else CDDM/pssms_upper.parquet)."""
    return _read_named_data(cls, "cddm_upper")

@patch(cls_method=True)
def cddm_LO(cls: Data) -> pd.DataFrame:
    """CDDM log-odds against the ks_dataset background: log2(freq) - log2(ks_STY background).

    The background pools every KS-dataset site with an S/T/Y acceptor, deduplicated on the all-uppercase
    site sequence; see ks_background(). Cells where freq == 0 or bg == 0 have undefined odds and are NaN.
    Regenerated by nbs/data_09_cddm_log_odds.py.
    """
    return _read_named_data(cls, "cddm_LO")

@patch(cls_method=True)
def cddm_LO_upper(cls: Data) -> pd.DataFrame:
    """All-uppercase CDDM log-odds against the ks_STY_upper background (see cddm_LO)."""
    return _read_named_data(cls, "cddm_LO_upper")

@patch(cls_method=True)
def pssm(cls: Data, name: str) -> pd.DataFrame:
    """Read a consolidated flat PSSM parquet (one row per kinase) from the repo's nbs/pssm/ folder.

    e.g. Data.pssm('pspa_norm'), Data.pssm('pspa_enrich'), Data.pssm('cddm_freq'), Data.pssm('cddm_logodds'),
    Data.pssm('mlp_attr'), Data.pssm('sd_ptyrvar_p2'). These are (re)built by nbs/plot_all_logo_heatmaps.py (§0
    build_pssm), so this always reads the latest on disk (call Data.clear_cache() after a rebuild)."""
    path = Path(__file__).resolve().parents[1] / "nbs" / "pssm" / f"{name}.parquet"
    if not path.exists():
        avail = sorted(p.stem for p in path.parent.glob("*.parquet")) if path.parent.exists() else []
        raise FileNotFoundError(f"No pssm '{name}' at {path}. Available: {avail}")
    return _read_file(str(path))

@patch(cls_method=True)
def aa_info(cls: Data) -> pd.DataFrame:
    """Get amino acid information."""
    return _read_named_data(cls, "aa_info")

@patch(cls_method=True)
def aa_rdkit(cls: Data) -> pd.DataFrame:
    """Get RDKit representations of amino acids."""
    return _read_named_data(cls, "aa_rdkit")

@patch(cls_method=True)
def aa_morgan(cls: Data) -> pd.DataFrame:
    """Get Morgan fingerprint representations of amino acids."""
    return _read_named_data(cls, "aa_morgan")

@patch(cls_method=True)
def cptac_ensembl_site(cls: Data) -> pd.DataFrame:
    """Get CPTAC dataset with unique EnsemblProteinID+site."""
    return _read_named_data(cls, "cptac_ensembl_site")

@patch(cls_method=True)
def cptac_unique_site(cls: Data) -> pd.DataFrame:
    """Get CPTAC dataset with unique site sequences."""
    return _read_named_data(cls, "cptac_unique_site")

@patch(cls_method=True)
def cptac_gene_site(cls: Data) -> pd.DataFrame:
    """Get CPTAC dataset with unique Gene+site."""
    return _read_named_data(cls, "cptac_gene_site")

@patch(cls_method=True)
def psp_human_site(cls: Data) -> pd.DataFrame:
    """Get PhosphoSitePlus human dataset (Gene+site)."""
    return _read_named_data(cls, "psp_human_site")

@patch(cls_method=True)
def ochoa_site(cls: Data) -> pd.DataFrame:
    """Get phosphoproteomics dataset from Ochoa et al."""
    return _read_named_data(cls, "ochoa_site")

@patch(cls_method=True)
def combine_site_psp_ochoa(cls: Data) -> pd.DataFrame:
    """
    Get the combined dataset from Ochoa and PhosphoSitePlus.
    """
    return _read_named_data(cls, "combine_site_psp_ochoa")

@patch(cls_method=True)
def combine_site_phosphorylated(cls: Data) -> pd.DataFrame:
    """
    Get the combined phosphorylated dataset from Ochoa and PhosphoSitePlus.
    """
    return _read_named_data(cls, "combine_site_phosphorylated")

@patch(cls_method=True)
def human_site(cls: Data) -> pd.DataFrame:
    """
    Get the combined phosphorylated dataset from Ochoa and PhosphoSitePlus (20-length version).
    """
    return _read_named_data(cls, "human_site")

@patch(cls_method=True)
def reactome_pathway_lo(cls: Data) -> pd.DataFrame:
    """
    Get lowest reactome pathways with Uniprot ID as identifier.
    """
    return _read_named_data(cls, "reactome_pathway_lo")

@patch(cls_method=True)
def reactome_pathway(cls: Data) -> pd.DataFrame:
    """
    Get all level reactome pathways with Uniprot ID as identifier.
    """
    path_all = _read_named_data(cls, "reactome_pathway")
    # path_lo = self.reactome_pathway_lo
    # path_all['lowest'] = path_all.reactome_id.isin(path_lo.reactome_id).astype(int)
    return path_all

def _read_cptac(
    cancer: str,  # CPTAC cancer type
    is_Tumor: bool = True,  # Tumor tissue or normal tissue
    is_KB: bool = False,  # LinkedOmicsKB protein-site identifiers instead of gene-site
) -> pd.DataFrame:
    "Fetch the CPTAC phosphoproteomics table for one cancer type."

    sample_type = "Tumor" if is_Tumor else "Normal"
    id_url = f"https://zenodo.org/records/8196130/files/bcm-{cancer.lower()}-mapping-gencode.v34.basic.annotation-mapping.txt.gz"
    data_url = f"https://cptac-pancancer-data.s3.us-west-2.amazonaws.com/data_freeze_v1.2_reorganized/{cancer.upper()}/{cancer.upper()}_phospho_site_abundance_log2_reference_intensity_normalized_{sample_type}.txt"

    ref = pd.read_csv(id_url, compression="gzip", sep="\t")[["protein", "gene", "gene_name"]]
    ref = ref.drop_duplicates().reset_index(drop=True)

    try: raw = pd.read_csv(data_url, sep="\t")
    except Exception as e: raise RuntimeError(f"Failed to load CPTAC data for {cancer}: {e}") from e

    info = pd.DataFrame(
        {
            "gene": raw.idx.str.split("|").str[0],
            "site": raw.idx.str.split("|").str[2],
            "site_seq": raw.idx.str.split("|").str[3],
        }
    )

    print(f"the {cancer} dataset length is: {info.shape[0]}")

    info = info.merge(ref, "left")
    print(f"after id mapping, the length is {info.shape[0]}")
    print(f"{info.gene_name.isna().sum()} sites does not have a mapped gene name")

    info["gene_site"] = info["gene_name"] + "_" + info["site"]
    info["protein_site"] = info["protein"].str.split(".").str[0] + "_" + info["site"]

    info = info.drop_duplicates(subset="protein_site" if is_KB else "gene_site").reset_index(drop=True)
    print(f"after removing duplicates of protein_site, the length is {info.shape[0]}")
    return info

class CPTAC:
    "A class for fetching CPTAC phosphoproteomics data."
    pass

@patch(cls_method=True)
def list_cancer(cls: CPTAC) -> list[str]:
    "List available CPTAC cancer type"
    return ['HNSCC','GBM','COAD','CCRCC','LSCC','BRCA','UCEC','LUAD','PDAC','OV']

@patch(cls_method=True)
def get_id(
    cls: CPTAC,  # Patched class receiver
    cancer_type: str,  # CPTAC cancer type
    is_Tumor: bool = True,  # Tumor tissue or normal tissue
    is_KB: bool = False,  # LinkedOmicsKB protein-site identifiers instead of gene-site
) -> pd.DataFrame:
    "Get CPTAC phosphorylation site information for one cancer type."
    if cancer_type not in cls.list_cancer():
        raise ValueError("cancer type is not included, check available cancer types from CPTAC.list_cancer()")
    return _read_file(cancer_type, is_Tumor, is_KB)
