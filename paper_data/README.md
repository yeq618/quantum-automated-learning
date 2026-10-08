# Data of the figures in the paper

Outputs of the runs shown in the paper, made with the default settings of the
scripts and `--seed 5`.

```
python scripts/make_figures.py --source paper
```

draws all panels from these files. Rerunning the scripts with the default seed
reproduces them.

| Folder | Figure | Command |
|---|---|---|
| `fashion_mnist/` | Fig. 2b–d | `simulate.py --dataset fashion_mnist` |
| `noisy/` | Fig. 2e | `noisy.py` |
| `reusability/` | Fig. 2f | `reusability.py` |
| `mnist/` | Supplementary Fig. 4 | `simulate.py --dataset mnist` |
| `aa/` | Supplementary Fig. 5 | `simulate.py --dataset aa` |
| `cim/` | Supplementary Fig. 6 | `simulate.py --dataset cim` |
| `spectrum_scaling/` | Supplementary Fig. 7 | `spectrum_scaling.py` |
| `fashion_mnist_large_lr/` | Supplementary Fig. 8 | `simulate.py --dataset fashion_mnist --lr 0.8 --steps 10 --report-interval 1 --tag fashion_mnist_large_lr` |

## File formats

- `train.csv`: one row per report. Columns are `step`, `test accuracy`,
  `train accuracy` (1 − ⟨ψ|H_S|ψ⟩ for the training set), `success probability`
  (product of all post-selection probabilities so far) and `data0` … `data499`,
  the probability of a correct prediction for each test sample.
- `eigenvalues.npy`: eigenvalues of H_S for the training set, rounded to 6
  decimals. `spectrum_scaling/eigenvalues_n<n>.npy` holds the same for n qubits,
  and `meta_n<n>.json` lists the training samples.
- `noisy/noise_rate<rate>.csv`: `step`, `accuracy` (on the training set) and
  `success probability`. `<rate>` is the two-qubit depolarizing rate.
- `reusability/train.csv`: `step`, `train loss` and `is correct`
  (1 correct prediction, 0 wrong prediction, 2 training step).
  `summary.json` gives the NSR statistics.
- `config.json`: the arguments of each run.
