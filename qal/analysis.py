"""Post-processing of the training logs: majority vote and state reusability."""
import math

import numpy as np

TRIALS = (np.inf, 29, 1)


def majority_vote(p, num_trials):
    """Probability that a majority vote over num_trials independent runs is correct.

    p is the probability that a single run predicts the correct label. Ties
    (possible for even num_trials) are broken uniformly at random. With
    infinitely many runs the vote is correct whenever p > 1/2.
    """
    if num_trials == np.inf:
        return float(p > 0.5)
    p_succ = 0
    for i in range(num_trials // 2 + 1):
        p_i = math.comb(num_trials, i) * (1 - p)**i * p**(num_trials - i)
        if 2 * i == num_trials:
            p_i = p_i / 2
        p_succ += p_i
    return p_succ


def add_majority_vote(df, trials=TRIALS):
    """Add a column '<k> trials' with the test accuracy of a k-run majority vote.

    The per-sample probabilities of a correct prediction are in the columns
    data0, data1, ... of the training log written by scripts/simulate.py.
    """
    data_columns = [c for c in df.columns if c.startswith('data')]
    for trial in trials:
        df[f'{trial} trials'] = df.loc[:, data_columns].map(lambda p: majority_vote(p, trial)).mean(axis=1)
    return df


def reusability_summary(df):
    """Statistics of the number of steps to recover (NSR) after each prediction.

    A recovery period runs from one prediction to the next. It is grouped by
    the outcome of the prediction that starts it. The initial training from
    the random state is counted as following a wrong prediction, as in the
    paper. The outcome of the last prediction has no recovery period.
    """
    predictions = df[df['is correct'] != 2.0]
    predict_steps = [0] + list(predictions['step'])
    recovery = np.array([predict_steps[i + 1] - predict_steps[i] for i in range(len(predict_steps) - 1)])
    previous = np.array([0] + list(predictions['is correct'])[:-1])
    wrong = np.where(previous == 0)[0]
    correct = np.where(previous == 1)[0]
    return {
        'predictions': len(predictions),
        'nsr_after_wrong': float(recovery[wrong].mean()),
        'nsr_after_correct': float(recovery[correct].mean()),
        'fraction_wrong': len(wrong) / len(previous),
        'fraction_correct': len(correct) / len(previous),
    }
