"""Datasets used in the paper and the on-disk cache of the matrices H_x."""
import abc
import multiprocessing as mp
from pathlib import Path

import numpy as np
from skimage.transform import resize

from .paths import DATA_DIR
from .quantum_basics import AubryAndreHamiltonian, ClusteringIsingHamiltonian


class Dataset(abc.ABC):
    @abc.abstractmethod
    def get_sample(self, idx):
        pass

    @abc.abstractmethod
    def get_random_sample(self):
        pass


class ClassicalDataset(Dataset):
    """Feature vectors with integer labels. indices[i] is the position of sample i in the source file."""

    def __init__(self, data, labels, indices=None):
        self.data = data
        self.labels = labels
        self.indices = np.arange(len(data)) if indices is None else indices
        self.len = len(data)

    def get_sample(self, idx):
        return self.data[idx], self.labels[idx]

    def get_random_sample(self):
        idx = np.random.randint(self.len)
        return self.get_sample(idx)

    def split(self, sizes):
        """Shuffle once and cut the shuffled data into consecutive pieces of the given sizes."""
        idx = np.random.permutation(self.len)
        data, labels, indices = self.data[idx], self.labels[idx], self.indices[idx]
        datasets = []
        start = 0
        for size in sizes:
            part = slice(start, start + size)
            datasets.append(ClassicalDataset(data[part], labels[part], indices[part]))
            start += size
        return tuple(datasets)

    def describe(self):
        return {'source_indices': self.indices.tolist(), 'labels': self.labels.tolist()}


def batch_resize(images, output_shape):
    resized = np.zeros((images.shape[0], *output_shape), dtype=np.float32)
    for i in range(images.shape[0]):
        resized[i] = resize(images[i], output_shape, mode='reflect', anti_aliasing=True)
    return resized


class BinaryImageDataset(ClassicalDataset):
    """Two classes of 28x28 images, downsampled to size x size.

    Pixel values are scaled to [0, 1], each image is resized with
    anti-aliasing and flattened, and every vector is normalized to unit
    2-norm. Images of classes[0] get label 0 and images of classes[1] label 1.
    """
    file_name = None

    def __init__(self, size, classes=(1, 9)):
        self.size = size
        with np.load(DATA_DIR / self.file_name) as f:
            images, labels = f['images'], f['labels']
        idx = np.where((labels == classes[0]) | (labels == classes[1]))[0]
        images = images[idx]
        labels = (labels[idx] == classes[1]).astype('int32')
        images = images / 255.0
        images = batch_resize(images, (size, size))
        images = images.reshape(len(labels), -1)
        images = images / np.linalg.norm(images, axis=1, keepdims=True)
        super().__init__(images, labels, indices=idx)


class MnistDataset(BinaryImageDataset):
    """MNIST digits "1" (label 0) and "9" (label 1)."""
    file_name = 'mnist.npz'


class FashionMnistDataset(BinaryImageDataset):
    """Fashion-MNIST classes "trouser" (1, label 0) and "ankle boot" (9, label 1)."""
    file_name = 'fashion_mnist.npz'


class HamiltonianDataset(Dataset):
    """Hamiltonians H(p) for a list of parameters p, labelled by the phase of H(p)."""

    def __init__(self, parameters, hamiltonian_map, label_map):
        self.parameters = parameters
        self.hamiltonian_map = hamiltonian_map
        self.label_map = label_map
        self.len = len(parameters)

    def get_sample(self, idx):
        return self.hamiltonian_map(self.parameters[idx]), self.label_map(self.parameters[idx])

    def get_random_sample(self):
        idx = np.random.randint(self.len)
        return self.get_sample(idx)

    def split(self, sizes):
        """Shuffle once and cut the shuffled parameters into consecutive pieces of the given sizes."""
        idx = np.random.permutation(self.len)
        parameters = self.parameters[idx]
        datasets = []
        start = 0
        for size in sizes:
            datasets.append(HamiltonianDataset(parameters[start:start + size], self.hamiltonian_map, self.label_map))
            start += size
        return tuple(datasets)

    def describe(self):
        return {'parameters': self.parameters.tolist(), 'labels': [self.label_map(p) for p in self.parameters]}


class AubryAndreHamiltonianDataset(HamiltonianDataset):
    """Aubry-Andre chains with disorder V. Label 0: delocalized (V < 2), label 1: localized (V > 2)."""

    def __init__(self, num_qubits, parameters):
        super().__init__(parameters,
                         lambda V: AubryAndreHamiltonian(num_qubits, V),
                         lambda V: 0 if V < 2.0 else 1)


class ClusteringIsingModelDataset(HamiltonianDataset):
    """Cluster-Ising chains with coupling h. Label 0: SPT phase (h < 1), label 1: antiferromagnetic (h > 1)."""

    def __init__(self, num_qubits, parameters):
        super().__init__(parameters,
                         lambda h: ClusteringIsingHamiltonian(num_qubits, h),
                         lambda h: 0 if h < 1.0 else 1)


def _save_H_worker(args):
    X, y, converter, path = args
    np.save(path, converter.convert(X, y))


def save_dataset_H(dataset, converter, path, num_workers):
    """Save H_x of every sample as H_{i}.npy, plus their average H_S and its spectrum.

    Each H_x is a dense 2^n x 2^n complex matrix (16 MB for n = 10).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    args = [(*dataset.get_sample(i), converter, path / f'H_{i}.npy') for i in range(dataset.len)]
    with mp.Pool(num_workers) as pool:
        pool.map(_save_H_worker, args)
    dim = 2**converter.num_qubits
    H_avg = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dataset.len):
        H_avg += np.load(path / f'H_{i}.npy')
    H_avg = (H_avg + H_avg.conj().T) / 2
    H_avg /= dataset.len
    np.save(path / 'H_average.npy', H_avg)
    eigs = np.round(np.linalg.eigvalsh(H_avg).real, 6)
    np.save(path / 'eigenvalues.npy', eigs)


class LoadingDataset(Dataset):
    """Cached matrices H_x, loaded from disk when needed."""

    def __init__(self, paths):
        self.paths = list(paths)
        self.len = len(self.paths)

    def get_sample(self, idx):
        return np.load(self.paths[idx])

    def get_random_sample(self):
        idx = np.random.randint(self.len)
        return self.get_sample(idx)
