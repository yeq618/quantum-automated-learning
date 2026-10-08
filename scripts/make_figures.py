"""Draw the figures of the paper at their final print size.

    python scripts/make_figures.py --source paper   # from paper_data/ (the runs shown in the paper)
    python scripts/make_figures.py                  # from results/ (your own runs)

Writes to figures/<source>/:
  fig2_bf.pdf  Fig. 2b-f (180 mm wide; panel a is a schematic drawn separately)
  ed1.pdf ... ed5.pdf  Extended Data Figs. 1-5 (ED Figs. 4 and 5 are one column, 88 mm, wide)
  summary.json  the main numbers
Missing inputs are skipped. Spectra are those of H_S for the training set,
with the sorted spectrum averaged with its mirror image E -> 1 - E.
--raw-spectrum skips the averaging.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qal import plotting
from qal.analysis import add_majority_vote
from qal.paths import FIGURES_DIR, PAPER_DATA_DIR, RESULTS_DIR

MM = plotting.MM


def figure_mm(width, height):
    return plt.figure(figsize=(width * MM, height * MM))


def axes_mm(fig, left, bottom, width, height):
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def fit_to_width(fig, ax, width_mm, margin_mm=0.5, rounds=6, pad_mm=40):
    """Resize the figure and the axes so that everything drawn spans width_mm.

    Used for the 3D panel. The extent is measured on a rendering of the figure,
    because the bounding box that matplotlib reports for 3D axes misses part of
    the z label. Text sizes stay fixed while the axes box is scaled.
    """
    pos = ax.get_position()
    W, H = fig.get_size_inches() / MM
    w, h = pos.width * W, pos.height * H
    dpi = fig.dpi
    fig.set_dpi(400)
    for _ in range(rounds):
        # draw the axes with free space around it and find the extent of the ink
        cw, ch = w + 2 * pad_mm, h + 2 * pad_mm
        fig.set_size_inches(cw * MM, ch * MM)
        ax.set_position([pad_mm / cw, pad_mm / ch, w / cw, h / ch])
        fig.canvas.draw()
        ink = np.asarray(fig.canvas.buffer_rgba())[..., :3].min(axis=2) < 250
        rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
        px = fig.dpi * MM
        x0, x1 = cols[0] / px, (cols[-1] + 1) / px
        y0, y1 = ch - (rows[-1] + 1) / px, ch - rows[0] / px
        scale = (width_mm - 2 * margin_mm) / (x1 - x0)
        if abs(scale - 1) < 1e-3:
            break
        w, h = w * scale, h * scale
    height = y1 - y0 + 2 * margin_mm
    fig.set_size_inches(width_mm * MM, height * MM)
    ax.set_position([(margin_mm + pad_mm - x0) / width_mm, (margin_mm + pad_mm - y0) / height, w / width_mm, h / height])
    fig.set_dpi(dpi)


def load_training(folder):
    path = folder / 'train.csv'
    return add_majority_vote(pd.read_csv(path)) if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', choices=['results', 'paper'], default='results')
    parser.add_argument('--out', type=Path, help='output folder (default: figures/<source>)')
    parser.add_argument('--raw-spectrum', action='store_true', help='plot the spectra without mirror averaging')
    args = parser.parse_args()
    src = PAPER_DATA_DIR if args.source == 'paper' else RESULTS_DIR
    out = args.out or FIGURES_DIR / args.source
    out.mkdir(parents=True, exist_ok=True)
    symmetrize = not args.raw_spectrum
    plotting.setup_style()
    summary = {}

    # Fig. 2b-f: two rows of 60 mm cells, panel f spans two cells. Between the rows
    # there is room for the x labels of the top row and the panel letters below them.
    fashion = load_training(src / 'fashion_mnist')
    noise_files = sorted((src / 'noisy').glob('noise_rate*.csv'), key=lambda p: float(p.stem[len('noise_rate'):]))
    reuse = src / 'reusability' / 'train.csv'
    if fashion is not None and noise_files and reuse.exists():
        fig = figure_mm(180, 101.5)
        top, bottom = 58.5, 9
        axes = {
            'b': axes_mm(fig, 11, top, 46, 38), 'c': axes_mm(fig, 71, top, 46, 38), 'd': axes_mm(fig, 131, top, 46, 38),
            'e': axes_mm(fig, 11, bottom, 46, 38), 'f': axes_mm(fig, 71, bottom, 106, 38),
        }
        plotting.training(axes['b'], fashion)
        plotting.tradeoff(axes['c'], fashion)
        plotting.spectrum(axes['d'], np.load(src / 'fashion_mnist' / 'eigenvalues.npy'), symmetrize)
        dfs = {float(p.stem[len('noise_rate'):]): pd.read_csv(p) for p in noise_files}
        plotting.noise(axes['e'], dfs)
        summary['reusability'] = plotting.reusability(axes['f'], pd.read_csv(reuse))
        for letter, ax in axes.items():
            plotting.panel_label(fig, ax, letter, dx_mm=-10)
        fig.savefig(out / 'fig2_bf.pdf')
        plt.close(fig)
        summary['fashion_mnist'] = final_accuracies(fashion)
        summary['noisy'] = {f'rate {rate}': {'final accuracy': float(df['accuracy'].iloc[-1])} for rate, df in dfs.items()}
    else:
        print('skipped Fig. 2b-f: missing fashion_mnist, noisy or reusability results')

    # Extended Data Figs. 1-3: one row of three 60 mm cells
    for name, number in (('mnist', 1), ('aa', 2), ('cim', 3)):
        df = load_training(src / name)
        if df is None:
            print(f'skipped Extended Data Fig. {number}: no {src / name / "train.csv"}')
            continue
        fig = figure_mm(180, 52)
        axes = [axes_mm(fig, 11 + 60 * i, 9, 46, 38) for i in range(3)]
        plotting.training(axes[0], df)
        plotting.tradeoff(axes[1], df)
        plotting.spectrum(axes[2], np.load(src / name / 'eigenvalues.npy'), symmetrize)
        for letter, ax in zip('abc', axes):
            plotting.panel_label(fig, ax, letter, dx_mm=-10)
        fig.savefig(out / f'ed{number}.pdf')
        plt.close(fig)
        summary[name] = final_accuracies(df)

    # Extended Data Fig. 4: spectrum versus number of qubits
    scaling = sorted((src / 'spectrum_scaling').glob('eigenvalues_n*.npy'))
    if scaling:
        eigenvalues = {int(p.stem.split('_n')[1]): np.load(p) for p in scaling}
        fig = figure_mm(88, 88)
        ax = plotting.spectrum_scaling(fig, [0.0, 0.0, 1.0, 1.0], eigenvalues, symmetrize)
        fit_to_width(fig, ax, 88)
        fig.savefig(out / 'ed4.pdf')
        plt.close(fig)
    else:
        print('skipped Extended Data Fig. 4: no spectrum_scaling eigenvalues')

    # Extended Data Fig. 5: one column wide
    df = load_training(src / 'fashion_mnist_large_lr')
    if df is not None:
        fig = figure_mm(88, 64)
        ax = axes_mm(fig, 12, 9, 72, 52)
        plotting.training(ax, df, max_step=None, xticks_every_step=True)
        fig.savefig(out / 'ed5.pdf')
        plt.close(fig)
        summary['fashion_mnist_large_lr'] = final_accuracies(df, step=None)
    else:
        print('skipped Extended Data Fig. 5: no fashion_mnist_large_lr results')

    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f'figures written to {out}')


def final_accuracies(df, step=100):
    """Accuracies at the last plotted step (step 100 for the main training panels)."""
    row = df.iloc[-1] if step is None else df[df['step'] <= step].iloc[-1]
    return {'step': int(row['step']), 'inf trials': row['inf trials'], '29 trials': row['29 trials'],
            '1 trial': row['1 trials'], 'training': row['train accuracy'],
            'success probability': row['success probability']}


if __name__ == '__main__':
    main()
