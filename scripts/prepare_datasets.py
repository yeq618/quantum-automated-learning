"""Rebuild data/mnist.npz and data/fashion_mnist.npz from the original IDX files.

Each folder must contain the four files of the official release
(train-images-idx3-ubyte, train-labels-idx1-ubyte, t10k-images-idx3-ubyte,
t10k-labels-idx1-ubyte), gzipped or not. The training and test sets are
concatenated (60,000 + 10,000 images) and saved with the arrays images
(uint8, 70000 x 28 x 28) and labels. Every other processing step (choice of
classes, downsampling, normalization) happens in qal/dataset.py.

    python scripts/prepare_datasets.py --mnist path/to/mnist --fashion-mnist path/to/fashion-mnist
"""
import argparse
import gzip
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from qal.paths import DATA_DIR

FILES = ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
# MD5 checksums of the uncompressed files from which the data of the paper were built
MD5 = {
    'mnist': {
        'train-images-idx3-ubyte': '6bbc9ace898e44ae57da46a324031adb',
        'train-labels-idx1-ubyte': 'a25bea736e30d166cdddb491f175f624',
        't10k-images-idx3-ubyte': '2646ac647ad5339dbf082846283269ea',
        't10k-labels-idx1-ubyte': '27ae3e4e09519cfbb04c329615203637',
    },
    'fashion_mnist': {
        'train-images-idx3-ubyte': 'f4a8712d7a061bf5bd6d2ca38dc4d50a',
        'train-labels-idx1-ubyte': '9018921c3c673c538a1fc5bad174d6f9',
        't10k-images-idx3-ubyte': '8181f5470baa50b63fa0f6fddb340f0a',
        't10k-labels-idx1-ubyte': '15d484375f8d13e6eb1aabb0c3f46965',
    },
}


def read_idx(folder, name):
    for path in (folder / name, folder / f'{name}.gz'):
        if path.exists():
            data = path.read_bytes()
            return gzip.decompress(data) if path.suffix == '.gz' else data
    raise FileNotFoundError(f'{name} or {name}.gz not found in {folder}')


def build(name, folder, out_dir):
    raw = {f: read_idx(folder, f) for f in FILES}
    for f, data in raw.items():
        md5 = hashlib.md5(data).hexdigest()
        if md5 != MD5[name][f]:
            print(f'warning: {folder / f} differs from the file used for the paper (md5 {md5})')
    images = np.concatenate([np.frombuffer(raw[f], np.uint8, offset=16).reshape(-1, 28, 28)
                             for f in ('train-images-idx3-ubyte', 't10k-images-idx3-ubyte')])
    labels = np.concatenate([np.frombuffer(raw[f], np.uint8, offset=8)
                             for f in ('train-labels-idx1-ubyte', 't10k-labels-idx1-ubyte')])
    np.savez_compressed(out_dir / f'{name}.npz', images=images, labels=labels)
    print(f'{out_dir / f"{name}.npz"}: {images.shape[0]} images')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mnist', type=Path, help='folder with the MNIST IDX files')
    parser.add_argument('--fashion-mnist', type=Path, help='folder with the Fashion-MNIST IDX files')
    parser.add_argument('--out', type=Path, default=DATA_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mnist:
        build('mnist', args.mnist, args.out)
    if args.fashion_mnist:
        build('fashion_mnist', args.fashion_mnist, args.out)


if __name__ == '__main__':
    main()
