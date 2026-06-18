# Rank Intervals for Leaderboards

<p align="center">
  <a href="https://arxiv.org/abs/2606.08679"><img src="https://img.shields.io/badge/arXiv-2606.08679-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" height="40"/></a>
</p>

---

Official repository for the paper: **Rank Intervals for Leaderboards: A Hierarchical Framework for Model Evaluation**.

This repository contains the code to construct task-level rank confidence intervals (CIs) from pairwise comparisons and aggregate them into leaderboard-level rank prediction intervals (PIs) using a distribution-free conformal approach.

---

## Installation

Create a virtual environment (recommended) and install the package with its dependencies:

```bash
pip install -e .
```

Requires **Python 3.11+** (tested with 3.13).

---

## Synthetic data simulations

The synthetic data generation process is implemented in `experiments/synthetic_data/synthetic_data_generation.py`.

A simulation example can be found in `experiments/synthetic_data/synthetic_data_ranking_example.ipynb`.

The synthetic data experiments presented in the paper were conducted by running code similar to the simulation example, using multiple repetitions and different configurations. Full details of the configuration setup are provided in Appendix E of the paper.

---

## Real-data applications


### Downloading benchmark data

Full TabArena and MMLU benchmark data are not included in the repository. Download them with
`experiments/applications/data/download_data.py` after installation.

The repository does include `experiments/applications/data/tabarena_leaderboard_v01.csv`, the
TabArena-v0.1 leaderboard reference table (Table A.1 in [TabArena: A Living Benchmark for Machine
Learning on Tabular Data](https://arxiv.org/abs/2506.16791)).

TabArena downloads require `git` on your PATH and a working C++ compiler (macOS: `xcode-select --install`).

**Run the downloader** (from the repository root):

```bash
python experiments/applications/data/download_data.py              # both TabArena and MMLU (default)
python experiments/applications/data/download_data.py tabarena     # TabArena only
python experiments/applications/data/download_data.py mmlu         # MMLU only
```

**Downloader outputs**

- TabArena: `experiments/applications/data/tabarena_results.csv`
- MMLU: `experiments/applications/data/mmlu_by_subject.pkl`

### Run the notebooks

Run the application notebooks from `experiments/applications/` (they load data from the local `data/` folder).

- TabArena: `experiments/applications/tabarena_example.ipynb`
- MMLU: `experiments/applications/mmlu_example.ipynb`

---

## Citation

```bibtex
@article{neuhof2026rank,
  title={Rank Intervals for Leaderboards: A Hierarchical Framework for Model Evaluation},
  author={Neuhof, Bitya and Benjamini, Yuval},
  journal={arXiv preprint arXiv:2606.08679},
  year={2026}
}
```

