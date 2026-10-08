"""State reusability on Fashion-MNIST (Fig. 2f).

Training continues for --steps steps. Whenever the training loss <psi|H_S|psi>
is at most --threshold, the next sample is used for a prediction (a
measurement that disturbs the state) instead of a training step. Results go to
results/reusability/: train.csv with columns step, train loss and is correct
(1 correct, 0 wrong, 2 no prediction), and summary.json with the NSR statistics.

Uses the same H_x cache as `simulate.py --dataset fashion_mnist` with the same seed.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from qal.analysis import reusability_summary
from qal.experiment import prepare_cache, set_seed
from qal.model import QuantumAutomatedLearningModel
from qal.paths import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--qubits', type=int, default=10)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--train-size', type=int, default=500)
    parser.add_argument('--test-size', type=int, default=500, help='only selects the cache shared with simulate.py')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--threshold', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    train, _, cache = prepare_cache('fashion_mnist', args.size, args.qubits, args.train_size, args.test_size,
                                    args.seed, args.workers)
    # restart the RNG so that the run does not depend on whether the cache already existed
    set_seed(args.seed)
    out = RESULTS_DIR / 'reusability'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps({**vars(args), 'cache': cache.name}, indent=1))

    model = QuantumAutomatedLearningModel(args.qubits, 2)
    H_train = np.load(cache / 'train' / 'H_average.npy')
    rows = []
    for i in range(args.steps + 1):
        loss = np.vdot(model.state, H_train @ model.state).real
        H = train.get_random_sample()
        if loss <= args.threshold:
            is_correct = int(model.predict(H))
        else:
            model.train(H, args.lr)
            is_correct = 2
        rows.append([i, loss, is_correct])
        if i % 1000 == 0 or i == args.steps:
            df = pd.DataFrame(rows, columns=['step', 'train loss', 'is correct'], dtype=float)
            df.to_csv(out / 'train.csv', index=False)
            print(f'step {i}: loss {loss:.4f}', flush=True)

    summary = reusability_summary(df)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(summary)


if __name__ == '__main__':
    main()
