"""Train the QAL model on one dataset and log the test and training accuracy.

Produces the data of Fig. 2b-d (fashion_mnist), Extended Data Figs. 1-3
(mnist, aa, cim) and Extended Data Fig. 5 (larger learning rate). Results go
to results/<tag>/: train.csv (one row per report), the spectrum of H_S for the
training set (eigenvalues.npy) and config.json.

    python scripts/simulate.py --dataset fashion_mnist
    python scripts/simulate.py --dataset fashion_mnist --lr 0.8 --steps 10 --report-interval 1 --tag fashion_mnist_large_lr
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from qal.experiment import DATASETS, prepare_cache, set_seed
from qal.model import QuantumAutomatedLearningModel, evaluation
from qal.paths import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--qubits', type=int, default=10)
    parser.add_argument('--size', type=int, default=10, help='images are downsampled to size x size pixels')
    parser.add_argument('--t', type=float, default=2.0, help='evolution time of the encoding (aa, cim)')
    parser.add_argument('--train-size', type=int, default=500)
    parser.add_argument('--test-size', type=int, default=500)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--report-interval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--tag', help='output folder under results/ (default: the dataset name)')
    args = parser.parse_args()

    set_seed(args.seed)
    train, test, cache = prepare_cache(args.dataset, args.size, args.qubits, args.train_size, args.test_size,
                                       args.seed, args.workers, t=args.t)
    out = RESULTS_DIR / (args.tag or args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(cache / 'train' / 'eigenvalues.npy', out / 'eigenvalues.npy')
    (out / 'config.json').write_text(json.dumps({**vars(args), 'cache': cache.name}, indent=1))

    model = QuantumAutomatedLearningModel(args.qubits, 2)
    H_train = np.load(cache / 'train' / 'H_average.npy')
    columns = ['step', 'test accuracy', 'train accuracy', 'success probability'] + [f'data{i}' for i in range(args.test_size)]
    rows = []
    for i in range(args.steps + 1):
        if i % args.report_interval == 0:
            accuracies = evaluation(model, test, args.workers)
            test_accuracy = np.mean(accuracies)
            train_accuracy = 1 - np.vdot(model.state, H_train @ model.state).real
            rows.append([i, test_accuracy, train_accuracy, model.succ_prob] + accuracies)
            pd.DataFrame(rows, columns=columns, dtype=float).to_csv(out / 'train.csv', index=False)
            print(f'step {i}: test {test_accuracy:.4f}, train {train_accuracy:.4f}, '
                  f'success probability {model.succ_prob:.3e}', flush=True)
        model.train(train.get_random_sample(), args.lr)


if __name__ == '__main__':
    main()
