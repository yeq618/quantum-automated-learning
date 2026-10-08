"""Shared set-up for the simulation scripts: seeding, datasets and the H_x cache."""
import json
import random
import time

import numpy as np

from .converter import HConverter
from .dataset import (AubryAndreHamiltonianDataset, ClusteringIsingModelDataset, FashionMnistDataset,
                      LoadingDataset, MnistDataset, save_dataset_H)
from .paths import CACHE_DIR, DATA_DIR
from .quantum_basics import CanonicalPVM, Hamiltonian

DATASETS = ('fashion_mnist', 'mnist', 'aa', 'cim')
IMAGE_DATASETS = ('fashion_mnist', 'mnist')


def set_seed(seed):
    """Seed the global NumPy and Python RNGs that the simulation code draws from."""
    random.seed(seed)
    np.random.seed(seed)


def load_cim_converter_hamiltonian():
    """The fixed 20-qubit Hamiltonian H_c used to encode cluster-Ising ground states (10 qubits)."""
    terms = json.loads((DATA_DIR / 'cim_converter_hamiltonian.json').read_text())
    return Hamiltonian(20, terms)


def build_dataset(name, size, num_qubits, t=2.0):
    """Return the full dataset and the converter that maps a sample to H_x."""
    PVM = CanonicalPVM(num_qubits, 2)
    if name == 'fashion_mnist':
        return FashionMnistDataset(size), HConverter(num_qubits, 'classical data', PVM)
    if name == 'mnist':
        return MnistDataset(size), HConverter(num_qubits, 'classical data', PVM)
    if name == 'aa':
        dataset = AubryAndreHamiltonianDataset(num_qubits, np.linspace(0, 4, 10000))
        return dataset, HConverter(num_qubits, 'hamiltonian evolution', PVM, t=t)
    if name == 'cim':
        if num_qubits != 10:
            raise ValueError('the cluster-Ising encoding Hamiltonian is defined for 10 qubits only')
        dataset = ClusteringIsingModelDataset(num_qubits, np.linspace(0, 2, 10000))
        converter = HConverter(num_qubits, 'hamiltonian ground state', PVM, t=t,
                               hamiltonian=load_cim_converter_hamiltonian())
        return dataset, converter
    raise ValueError(f'unknown dataset {name!r}')


def cache_dir(name, size, num_qubits, train_size, test_size, seed, t=2.0):
    detail = f'size{size}' if name in IMAGE_DATASETS else f't{t}'
    return CACHE_DIR / f'{name}_{num_qubits}q_{detail}_train{train_size}_test{test_size}_seed{seed}'


def prepare_cache(name, size, num_qubits, train_size, test_size, seed, num_workers, t=2.0):
    """Split the dataset and make sure H_x of every train and test sample is cached.

    Call set_seed(seed) first. The split is then the first draw from the
    global RNG, so a given seed always selects the same samples, whether or
    not the cache already exists. Returns the train and test LoadingDatasets
    and the cache directory.
    """
    dataset, converter = build_dataset(name, size, num_qubits, t)
    train, test = dataset.split([train_size, test_size])
    path = cache_dir(name, size, num_qubits, train_size, test_size, seed, t)
    meta_file = path / 'meta.json'
    if not meta_file.exists():
        print(f'computing H_x for {train_size + test_size} samples in {path}', flush=True)
        start = time.time()
        save_dataset_H(train, converter, path / 'train', num_workers)
        save_dataset_H(test, converter, path / 'test', num_workers)
        meta = {'dataset': name, 'qubits': num_qubits, 'seed': seed,
                'train': train.describe(), 'test': test.describe(),
                'seconds': round(time.time() - start, 1)}
        if name in IMAGE_DATASETS:
            meta['size'] = size
        else:
            meta['t'] = t
        meta_file.write_text(json.dumps(meta))

    def load(subset, n):
        return LoadingDataset([path / subset / f'H_{i}.npy' for i in range(n)])

    return load('train', train_size), load('test', test_size), path
