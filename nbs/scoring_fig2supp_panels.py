"""scoring_fig2supp_panels · Fig2 supplement (no num_kin <= 10 test restriction).

The same six panels as the main Figure 2, but computed on the FULL test set instead of the reliably
annotated num_kin <= 10 subset, so the reader can see whether the method ranking holds when the noisier
many-kinase sites are included. All drawing and data loading is reused from `scoring_fig2_panels.build`
with the num_kin cap turned off (nk_cap=None); only the output filename prefix differs.

Inputs   as scoring_fig2_panels (the persisted scoring_pairs + split/pool)
Outputs  fig/fig2supp{a..f}_*.svg, fig/fig2supp_method_legend.svg

Run:  python nbs/scoring_fig2supp_panels.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring_fig2_panels as f2


def main():
    f2.build(nk_cap=None, prefix='fig2supp')     # None = no num_kin cap: every test site


if __name__ == '__main__':
    main()
