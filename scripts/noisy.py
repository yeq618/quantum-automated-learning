"""Training under depolarizing gate noise (Fig. 2e).

Fashion-MNIST images are downsampled to 5x5 and encoded on 5 qubits so that
the density matrix can be simulated. Every one-qubit gate is followed by
depolarizing noise of strength rate/10 and every two-qubit gate by noise of
strength rate. The accuracy is evaluated on the training set. Results go to
results/noisy/noise_rate<rate>.csv.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from qal.dataset import FashionMnistDataset
from qal.experiment import set_seed
from qal.model import MixedQuantumAutomatedLearningModel, mixed_evaluation
from qal.paths import RESULTS_DIR
from qal.quantum_basics import DepolarizingNoiseModel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--qubits', type=int, default=5)
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--train-size', type=int, default=500)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--report-interval', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--noise-rates', type=float, nargs='+', default=[0.0, 0.001, 0.005],
                        help='two-qubit depolarizing rates')
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)
    train_dataset, = FashionMnistDataset(size=args.size).split([args.train_size])
    out = RESULTS_DIR / 'noisy'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=1))

    for rate in args.noise_rates:
        noise_model = DepolarizingNoiseModel(rate / 10, rate)
        model = MixedQuantumAutomatedLearningModel(args.qubits, 2, noise_model)
        rows = []
        for i in range(args.steps + 1):
            if i % args.report_interval == 0:
                accuracy = np.mean(mixed_evaluation(model, train_dataset, args.workers))
                rows.append([i, accuracy, model.succ_prob])
                pd.DataFrame(rows, columns=['step', 'accuracy', 'success probability'], dtype=float).to_csv(
                    out / f'noise_rate{rate}.csv', index=False)
                print(f'rate {rate}, step {i}: accuracy {accuracy:.4f}', flush=True)
            model.train(train_dataset.get_random_sample(), args.lr)
            # keep the state positive definite
            model.state += 1e-8 * np.eye(2**args.qubits)
            model.state = model.state / np.trace(model.state)


if __name__ == '__main__':
    main()
