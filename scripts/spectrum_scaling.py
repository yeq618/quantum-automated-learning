"""Spectrum of H_S for Fashion-MNIST on 5 to 12 qubits (Supplementary Fig. 7).

For n qubits the images are downsampled to n x n pixels. The training set is
drawn as in simulate.py, so with the same seed and n = 10 it contains the same
samples. H_S is averaged on the fly and per-sample matrices are never stored
(at n = 12 a single H_x takes 268 MB and about 25 s to compute). Results go to
results/spectrum_scaling/eigenvalues_n<n>.npy.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from qal.converter import HConverter
from qal.dataset import FashionMnistDataset
from qal.experiment import set_seed
from qal.paths import RESULTS_DIR
from qal.quantum_basics import CanonicalPVM

CHUNK = 5


def _partial_sum(args):
    samples, converter = args
    total = None
    for X, y in samples:
        H = converter.convert(X, y)
        total = H if total is None else total + H
    return total


def average_hamiltonian(dataset, converter, num_workers):
    """H_S = mean of H_x over the dataset. Chunks are summed in a fixed order, so the result does not depend on num_workers."""
    samples = [dataset.get_sample(i) for i in range(dataset.len)]
    chunks = [(samples[k:k + CHUNK], converter) for k in range(0, len(samples), CHUNK)]
    H_sum = None
    with mp.Pool(num_workers) as pool:
        for part in pool.imap(_partial_sum, chunks):
            H_sum = part if H_sum is None else H_sum + part
    H_avg = (H_sum + H_sum.conj().T) / 2
    return H_avg / dataset.len


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--qubits', type=int, nargs='+', default=list(range(5, 13)))
    parser.add_argument('--train-size', type=int, default=500)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()

    out = RESULTS_DIR / 'spectrum_scaling'
    out.mkdir(parents=True, exist_ok=True)
    for n in args.qubits:
        set_seed(args.seed)
        train, = FashionMnistDataset(size=n).split([args.train_size])
        converter = HConverter(n, 'classical data', CanonicalPVM(n, 2))
        start = time.time()
        eigenvalues = np.round(np.linalg.eigvalsh(average_hamiltonian(train, converter, args.workers)).real, 6)
        np.save(out / f'eigenvalues_n{n}.npy', eigenvalues)
        meta = {'qubits': n, 'seed': args.seed, 'seconds': round(time.time() - start, 1), 'train': train.describe()}
        (out / f'meta_n{n}.json').write_text(json.dumps(meta))
        print(f'n = {n}: {meta["seconds"]} s, lowest eigenvalue {eigenvalues[0]:.4f}', flush=True)


if __name__ == '__main__':
    main()
