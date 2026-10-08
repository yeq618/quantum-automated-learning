# Quantum automated learning: simulation code

Code for the numerical results in Q. Ye, S. Geng, Z. Han, W. Li, L.-M. Duan and
D.-L. Deng, *Quantum automated learning* ([arXiv:2502.05264](https://arxiv.org/abs/2502.05264)).

## Layout

```
qal/                    library
  quantum_basics.py     Pauli operators, model Hamiltonians, label projector, noise model
  converter.py          data encodings and the map x -> H_x
  dataset.py            datasets and the on-disk cache of H_x
  model.py              QAL model (pure state and noisy density matrix) and evaluation
  experiment.py         seeding, dataset construction, cache handling
  analysis.py           majority vote and reusability statistics
  plotting.py           figure panels
scripts/
  simulate.py           training runs (Fig. 2b-d, Supplementary Figs. 4-6 and 8)
  reusability.py        state reusability (Fig. 2f)
  noisy.py              training with depolarizing noise (Fig. 2e)
  spectrum_scaling.py   spectrum of H_S for 5 to 12 qubits (Supplementary Fig. 7)
  make_figures.py       draws all panels from results/ or paper_data/
  run_all.sh            runs everything with the settings of the paper
  prepare_datasets.py   rebuilds data/*.npz from the original MNIST and Fashion-MNIST files
data/                   MNIST, Fashion-MNIST, cluster-Ising encoding Hamiltonian
paper_data/             outputs of the runs shown in the paper
```

Generated files go to `cache/`, `results/` and `figures/`.

## Installation

Python 3.12 with

```
pip install -r requirements.txt
```

## Reproducing the figures

**From the stored data (seconds).** `paper_data/` holds the outputs of the
runs shown in the paper, all made with the default settings and `--seed 5`.

```
python scripts/make_figures.py --source paper
```

draws the panels into `figures/paper/`: `fig2_bf.pdf` (Fig. 2b–f) and
`supp_fig4.pdf` … `supp_fig8.pdf` (Supplementary Figs. 4–8), at their final
print size. Fig. 1 and Fig. 2a are schematics drawn separately and are not
produced by this code.

**Rerunning the simulations.** `bash scripts/run_all.sh` runs everything with
the settings of the paper (set `WORKERS` to the number of processes). The
scripts can also be run one by one:

| Figure | Command | Output in `results/` |
|---|---|---|
| Fig. 2b–d | `python scripts/simulate.py --dataset fashion_mnist` | `fashion_mnist/` |
| Fig. 2e | `python scripts/noisy.py` | `noisy/` |
| Fig. 2f | `python scripts/reusability.py` | `reusability/` |
| Supplementary Fig. 4 | `python scripts/simulate.py --dataset mnist` | `mnist/` |
| Supplementary Fig. 5 | `python scripts/simulate.py --dataset aa` | `aa/` |
| Supplementary Fig. 6 | `python scripts/simulate.py --dataset cim` | `cim/` |
| Supplementary Fig. 7 | `python scripts/spectrum_scaling.py` | `spectrum_scaling/` |
| Supplementary Fig. 8 | `python scripts/simulate.py --dataset fashion_mnist --lr 0.8 --steps 10 --report-interval 1 --tag fashion_mnist_large_lr` | `fashion_mnist_large_lr/` |

Then `python scripts/make_figures.py` draws the panels into `figures/results/`
and writes the main numbers to `summary.json`. With the default seed and the
package versions in `requirements.txt`, the reruns reproduce `paper_data/`. Every script lists its options
with `--help`. The defaults are the settings of the paper: 500 training and 500
test samples, learning rate 0.2, 10 qubits (5 for Fig. 2e), 100 training steps,
10,000 steps for Fig. 2f and 200 for Fig. 2e.

**Resources.** Each H_x is stored as a dense matrix of 16 MB, so the cache
takes 16 GB per dataset (63 GB in total). Approximate times with 16 worker
processes per job on a many-core server:

| Step | Time |
|---|---|
| H_x cache, 1000 samples (once per dataset and seed) | 2–2.5 min (images), 4 min (AA), 8.5 min (CIM) |
| Training run, 100 steps | 20 s |
| Fig. 2f, 10,000 steps | 1.5 min |
| Fig. 2e, 3 noise rates (20 workers) | 7.5 min |
| Supplementary Fig. 7 (48 workers) | 9 min, mostly the 12-qubit case |

Set `OMP_NUM_THREADS=1` (as `run_all.sh` does) when using many worker
processes. The cache location can be changed with `QAL_CACHE_DIR`.

## Notes

- **Random seeds.** Each script takes `--seed` (default 5, used for all
  figures of the paper). With the same seed and sizes, the train/test split and
  all results are identical, independent of the number of workers. The scripts
  with a given seed share one cache, so Fig. 2b–d, Fig. 2f and Supplementary
  Fig. 8 use the same samples.
- **Spectra.** The spectrum panels show H_S of the training set. The sorted
  spectrum is averaged with its mirror image E → 1 − E. The raw spectra are
  already nearly symmetric (the largest change is 0.009), and `--raw-spectrum`
  plots them without the averaging.
- **Reusability statistics.** The number of steps to recover (NSR) is measured
  from one prediction to the next and grouped by the outcome of the first one.
  The initial training from the random state counts as following a wrong
  prediction. The pie chart and the averages in Fig. 2f use all 10,000 steps.
  The curve shows the steps up to two steps after the first correct
  prediction that follows a wrong one (166 steps with seed 5), so that a
  whole recovery from a wrong prediction is visible.
- **Encoding Hamiltonian for cluster-Ising states.** `data/cim_converter_hamiltonian.json`
  is a fixed random 20-qubit Hamiltonian (`RandomHamiltonian(10, 10, 100)` in
  `qal/quantum_basics.py`), defined for 10-qubit states.
- **Fonts.** The panels use Arial, regular and bold, with all text at 7 pt.
  Arial is not distributed here. Put the font files in `data/fonts/` or
  install them, otherwise a similar free font is used.

## Data

`data/mnist.npz` and `data/fashion_mnist.npz` contain the 70,000 images of
MNIST (LeCun et al., 1998) and Fashion-MNIST (Xiao et al., 2017, MIT license),
with arrays `images` (28×28, uint8) and `labels`. They are the training and
test sets of the original releases, concatenated in this order.
MNIST is included for convenience under the terms of its original release
(http://yann.lecun.com/exdb/mnist/), and Fashion-MNIST under the MIT license
(https://github.com/zalandoresearch/fashion-mnist).
`scripts/prepare_datasets.py` rebuilds them from the original IDX files and
checks them against the files used for the paper. All further processing
(classes 1 and 9, downsampling to n×n pixels with anti-aliasing, unit 2-norm)
is done in `qal/dataset.py`.

The Hamiltonian datasets need no external data. The Aubry-André and
cluster-Ising Hamiltonians are generated in `qal/quantum_basics.py` for 10,000
parameter values (V in [0, 4], h in [0, 2]), and the cluster-Ising ground
states are computed by exact diagonalization. The only stored ingredient is
the fixed random encoding Hamiltonian in `data/cim_converter_hamiltonian.json`.

## AI disclosure

The original simulation code was written by the authors. Claude (Anthropic),
used as a coding agent under the authors' direction, organized it into this
release. It restructured the code into the `qal` package and the scripts,
added the seeding, the documentation and the figure scripts, and checked that
the reruns reproduce the stored results in `paper_data/`.

## Citation

If you use this code, please cite the paper (Q. Ye, S. Geng, Z. Han, W. Li,
L.-M. Duan and D.-L. Deng, *Quantum automated learning*, arXiv:2502.05264) and
this archived version of the code (Zenodo, DOI to be added on release). See
also `CITATION.cff`.

## License

MIT, see `LICENSE`.
