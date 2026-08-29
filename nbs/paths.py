"""Shared data-pool paths for the nbs analysis scripts and notebooks.

raw/ out/ fig/ live at the nbs root and are shared across modules. Scripts import RAW/OUT/FIG
from here (absolute, so they run from any working directory); notebooks in module subfolders
reach them via the small header cell that cd's to nbs/.

The reference-dataset store is nbs/katlas_datasets/ — see kdata.py, not this module.
"""
from pathlib import Path
_NBS = Path(__file__).resolve().parent
RAW, OUT, FIG = _NBS / "raw", _NBS / "out", _NBS / "fig"
PSSM = _NBS / "pssm"
