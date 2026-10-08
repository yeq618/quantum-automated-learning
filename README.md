# Quantum automated learning: simulation code

Code for the numerical results in Q. Ye, S. Geng, Z. Han, W. Li, L.-M. Duan and
D.-L. Deng, *Quantum automated learning* ([arXiv:2502.05264](https://arxiv.org/abs/2502.05264)).

## What is simulated

Every training sample x with label y is turned into the Hamiltonian

    H_x = I - U(x)^† P_y U(x),

where U(x) encodes the sample (a parametrized circuit for images, real-time
evolution for Hamiltonians, or an encoded ground state) and P_y projects the
middle qubit onto the label. ⟨ψ|H_x|ψ⟩ is the probability of predicting the
wrong label. A training step applies the post-selected update
|ψ⟩ ← (I − η H_x)|ψ⟩ / ‖·‖ for a randomly drawn sample, starting from a random
computational basis state. The simulations store every H_x as a dense matrix
and track the state vector exactly (a density matrix with gate noise for
Fig. 2e).

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
  simulate.py           training runs (Fig. 2b-d, Extended Data Figs. 1-3 and 5)
  reusability.py        state reusability (Fig. 2f)
  noisy.py              training with depolarizing noise (Fig. 2e)
  spectrum_scaling.py   spectrum of H_S for 5 to 12 qubits (Extended Data Fig. 4)
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

draws the panels into `figures/paper/`.

**Rerunning the simulations.** `bash scripts/run_all.sh` runs everything with
the settings of the paper (set `WORKERS` to the number of processes). The
scripts can also be run one by one:

| Figure | Command | Output in `results/` |
|---|---|---|
| Fig. 2b–d | `python scripts/simulate.py --dataset fashion_mnist` | `fashion_mnist/` |
| Fig. 2e | `python scripts/noisy.py` | `noisy/` |
| Fig. 2f | `python scripts/reusability.py` | `reusability/` |
| Extended Data Fig. 1 | `python scripts/simulate.py --dataset mnist` | `mnist/` |
| Extended Data Fig. 2 | `python scripts/simulate.py --dataset aa` | `aa/` |
| Extended Data Fig. 3 | `python scripts/simulate.py --dataset cim` | `cim/` |
| Extended Data Fig. 4 | `python scripts/spectrum_scaling.py` | `spectrum_scaling/` |
| Extended Data Fig. 5 | `python scripts/simulate.py --dataset fashion_mnist --lr 0.8 --steps 10 --report-interval 1 --tag fashion_mnist_large_lr` | `fashion_mnist_large_lr/` |

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
| Extended Data Fig. 4 (48 workers) | 9 min, mostly the 12-qubit case |

Set `OMP_NUM_THREADS=1` (as `run_all.sh` does) when using many worker
processes. The cache location can be changed with `QAL_CACHE_DIR`.

## Notes

- **Random seeds.** Each script takes `--seed` (default 5, used for all
  figures of the paper). With the same seed and sizes, the train/test split and
  all results are identical, independent of the number of workers. The scripts
  with a given seed share one cache, so Fig. 2b–d, Fig. 2f and Extended Data
  Fig. 5 use the same samples.
- **Spectra.** The spectrum panels show H_S of the training set. The sorted
  spectrum is averaged with its mirror image E → 1 − E. The raw spectra are
  already nearly symmetric (the largest change is 0.009), and `--raw-spectrum`
  plots them without the averaging.
- **Reusability statistics.** The number of steps to recover (NSR) is measured
  from one prediction to the next and grouped by the outcome of the first one.
  The initial training from the random state counts as following a wrong
  prediction.
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
`scripts/prepare_datasets.py` rebuilds them from the original IDX files and
checks them against the files used for the paper. All further processing
(classes 1 and 9, downsampling to n×n pixels with anti-aliasing, unit 2-norm)
is done in `qal/dataset.py`.

The Hamiltonian datasets need no external data. The Aubry-André and
cluster-Ising Hamiltonians are generated in `qal/quantum_basics.py` for 10,000
parameter values (V in [0, 4], h in [0, 2]), and the cluster-Ising ground
states are computed by exact diagonalization. The only stored ingredient is
the fixed random encoding Hamiltonian in `data/cim_converter_hamiltonian.json`.

## Citation

See `CITATION.cff`.

## License

MIT, see `LICENSE`.
