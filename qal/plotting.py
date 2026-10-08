"""Figure panels of the paper, drawn at their final print size.

The style follows the Nature artwork guide: Arial (or a similar free font if
Arial is not available), text between 5 and 7 pt (all text here is 7 pt), and
TrueType fonts embedded in the PDF so that text stays editable. Call
setup_style() once before plotting. Arial is used if it is installed or placed
in data/fonts (it is not distributed with this code). The panel letters need
the bold face as well.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D

from .analysis import TRIALS, reusability_summary
from .paths import FONT_DIR

MM = 1 / 25.4  # inches per millimetre
COLOR1 = '#476EAC'
COLOR2 = (240 / 255, 141 / 255, 72 / 255)
COLOR3 = '#027272'
NOISE_COLORS = {0.0: COLOR1, 0.001: COLOR2, 0.005: COLOR3}
NOISE_MARKERS = {0.0: 'd', 0.001: 'o', 0.005: 's'}
TICK_PT, LABEL_PT, LEGEND_PT, PANEL_PT = 7, 7, 7, 7


def _first_available(families):
    for family in families:
        try:
            font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
            return family
        except ValueError:
            continue
    return 'DejaVu Sans'


def setup_style():
    if FONT_DIR.is_dir():
        for font in sorted(FONT_DIR.iterdir()):
            if font.suffix.lower() in ('.ttf', '.otf'):
                font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({
        'font.family': _first_available(['Arial', 'Liberation Sans', 'Arimo', 'Helvetica', 'DejaVu Sans']),
        'font.size': TICK_PT,
        'axes.labelsize': LABEL_PT,
        'xtick.labelsize': TICK_PT,
        'ytick.labelsize': TICK_PT,
        'legend.fontsize': LEGEND_PT,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'xtick.major.pad': 2,
        'ytick.major.pad': 2,
        'axes.labelpad': 2,
        'lines.linewidth': 1.0,
        'lines.markersize': 3,
        'patch.linewidth': 0.5,
        'legend.frameon': True,
        'legend.framealpha': 0.85,
        'legend.edgecolor': '0.6',
        'legend.borderpad': 0.3,
        'legend.labelspacing': 0.25,
        'legend.handlelength': 1.8,
        'legend.handletextpad': 0.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 600,
    })


def _trial_label(trial):
    return f'{trial} trials' if trial != 1 else '1 trial'


def panel_label(fig, ax, letter, dx_mm=-9.0, dy_mm=1.5):
    """Bold lowercase panel label at the top left of an axes, offset in millimetres."""
    x0, y1 = ax.get_position().x0, ax.get_position().y1
    w, h = fig.get_size_inches() / MM
    fig.text(x0 + dx_mm / w, y1 + dy_mm / h, letter, fontsize=PANEL_PT, fontweight='bold', va='bottom', ha='left')


def training(ax, df, max_step=100, xticks_every_step=False):
    """Majority-vote test accuracy and training accuracy versus step (Fig. 2b, Supplementary Figs. 4a-6a and 8)."""
    if max_step is not None:
        df = df[df['step'] <= max_step]
    styles = [('k', '', 0, '--'), (COLOR1, 'o', 2.5, '-'), (COLOR2, '', 0, '-')]
    for trial, (color, marker, ms, ls) in zip(TRIALS, styles):
        ax.plot(df['step'], df[f'{trial} trials'], label=_trial_label(trial), color=color, marker=marker,
                markersize=ms, linestyle=ls)
    ax.scatter(df['step'], df['train accuracy'], label='Training', color=COLOR3, zorder=10, marker='d', s=7, linewidths=0)
    ax.legend(loc='lower right')
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    if xticks_every_step:
        ax.set_xticks(np.arange(len(df)))


def tradeoff(ax, df, max_step=100):
    """Majority-vote test accuracy versus overall post-selection success probability (Fig. 2c, Supplementary Figs. 4b-6b)."""
    if max_step is not None:
        df = df[df['step'] <= max_step]
    styles = [('k', '', 0, '--'), (COLOR1, 'o', 2.5, '-'), (COLOR2, 'd', 2.5, '-')]
    for trial, (color, marker, ms, ls) in zip(TRIALS, styles):
        ax.plot(df['success probability'], df[f'{trial} trials'], label=_trial_label(trial), color=color,
                marker=marker, markersize=ms, linestyle=ls)
    ax.set_xscale('log')
    # decimal tick labels: exponents of 10^n labels would be smaller than the 5 pt minimum
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.{max(0, -int(np.floor(np.log10(v) + 1e-9)))}f}'))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.legend(loc='lower left')
    ax.set_xlabel('Success probability')
    ax.set_ylabel('Accuracy')


def symmetrize_spectrum(eigenvalues):
    """Average the sorted spectrum with its mirror image E -> 1 - E."""
    e = np.sort(eigenvalues)
    return (e + 1 - e[::-1]) / 2


def spectrum(ax, eigenvalues, symmetrize=True):
    """Histogram of the eigenvalues of H_S (Fig. 2d, Supplementary Figs. 4c-6c)."""
    e = symmetrize_spectrum(eigenvalues) if symmetrize else np.sort(eigenvalues)
    ax.hist(e, bins=21, histtype='stepfilled', edgecolor='k', facecolor='#EEF3F2', linewidth=0.8)
    ax.set_xlabel('Energy')
    ax.set_ylabel('Frequency')


def noise(ax, dfs):
    """Training accuracy under depolarizing noise, one curve per two-qubit noise rate (Fig. 2e)."""
    for rate, df in dfs.items():
        ax.plot(df['step'], df['accuracy'], label=f'{rate * 1000:.0f}‰', color=NOISE_COLORS.get(rate),
                marker=NOISE_MARKERS.get(rate, 'o'), markersize=2.5, markevery=2)
    ax.legend(loc='center right', bbox_to_anchor=(1, 0.42))
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.5, 1)


CORRECT_MARK = dict(marker='*', color='green', s=16, linewidths=0)
WRONG_MARK = dict(marker='x', color='red', s=9, linewidths=0.7)


def _first_full_recovery(df):
    """Step of the first correct prediction that follows a wrong one."""
    flags = df['is correct'].to_numpy()
    wrong = np.flatnonzero(flags == 0.0)
    if len(wrong):
        correct = np.flatnonzero(flags[wrong[0]:] == 1.0)
        if len(correct):
            return int(df['step'].iloc[wrong[0] + correct[0]])
    return int(df['step'].iloc[-1])


def _marker(x, y, transform, style):
    """One scatter marker drawn as a Line2D, for annotations."""
    return Line2D([x], [y], transform=transform, linestyle='none', marker=style['marker'], color=style['color'],
                  markersize=np.sqrt(style['s']), markeredgewidth=style['linewidths'])


def reusability(ax, df, steps_shown=None, threshold=0.15, min_label_steps=3, pie_radius_mm=10.5):
    """Training loss with predictions, NSR tick labels, pie chart and average NSR (Fig. 2f).

    The curve runs from step 0 to steps_shown. By default it ends two steps
    after the first correct prediction that follows a wrong one, so that a
    whole recovery from a wrong prediction is visible. Intervals shorter than
    min_label_steps keep their tick but get no label (at print size the label
    of a 2-step interval would touch its neighbours). The pie chart and the averages use the whole run. As in the
    original figure, the wedge of wrong predictions sits on the horizontal
    radius, and a dashed line continuing that radius separates the average
    NSR after wrong predictions (above) from the one after correct predictions
    (below). The markers next to these averages explain the prediction
    markers on the curve. Returns the statistics from reusability_summary.
    """
    summary = reusability_summary(df)
    if steps_shown is None:
        steps_shown = _first_full_recovery(df) + 2
    shown = df[df['step'] <= steps_shown]
    ax.plot(shown['step'], shown['train loss'], color='k', label='Training loss')
    ax.axhline(threshold, color='k', linestyle='--', linewidth=0.8, label='Threshold')
    for flag, style in ((1.0, CORRECT_MARK), (0.0, WRONG_MARK)):
        steps = shown.loc[shown['is correct'] == flag, 'step']
        ax.scatter(steps, np.full(len(steps), threshold - 0.02), zorder=5, **style)
    legend = ax.legend(loc='upper left')
    predict_steps = [0] + [int(step) for step in shown.loc[shown['is correct'] != 2.0, 'step']]
    midpoints = [(a + b) / 2 for a, b in zip(predict_steps[:-1], predict_steps[1:])]
    lengths = [b - a for a, b in zip(predict_steps[:-1], predict_steps[1:])]
    ax.set_xticks(predict_steps, [''] * len(predict_steps))
    ax.set_xticks(midpoints, [str(n) if n >= min_label_steps else '' for n in lengths], minor=True)
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.set_xlim(0, steps_shown)
    ax.set_xlabel('Number of steps to recover (NSR)')
    ax.set_ylabel('Training loss')

    # layout in millimetres, converted to axes fractions (mx, my per mm)
    renderer = ax.figure.canvas.get_renderer()
    box = ax.get_window_extent(renderer)
    W, H = ax.figure.get_size_inches() / MM
    mx, my = box.width / ax.figure.bbox.width * W, box.height / ax.figure.bbox.height * H
    mx, my = 1 / mx, 1 / my
    width = lambda artist: artist.get_window_extent(renderer).width / box.width

    # pie chart right of the legend, touching the top of the axes
    rx, ry = pie_radius_mm * mx, pie_radius_mm * my
    cx = (legend.get_window_extent(renderer).x1 - box.x0) / box.width + 2.0 * mx + rx
    cy = 1 - 1.0 * my - ry
    inset = ax.inset_axes([cx - rx, cy - ry, 2 * rx, 2 * ry])
    fractions = [summary['fraction_wrong'], summary['fraction_correct']]
    _, _, percents = inset.pie(fractions, autopct='%1.0f%%', startangle=0, colors=['#D7CAFB', '#D8F9C8'],
                               pctdistance=0.7, textprops={'fontsize': TICK_PT},
                               wedgeprops={'edgecolor': 'black', 'linewidth': 0.6})
    for fraction, text in zip(fractions, percents):
        if fraction > 0.5:  # the label of the large wedge sits nearer the centre
            x, y = text.get_position()
            text.set_position((0.65 * x, 0.65 * y))
    inset.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), aspect='equal')

    # averages right of the pie: wrong above the dashed line, correct below
    kw = dict(transform=ax.transAxes, fontsize=LEGEND_PT, va='center')
    averages = [ax.text(0, 0, f'Avg. NSR = {summary[key]:.2f}', ha='center', **kw)
                for key in ('nsr_after_wrong', 'nsr_after_correct')]
    x0 = cx + rx + 0.6 * mx
    x1 = x0 + max(width(t) for t in averages) + 1.5 * mx
    xc = (x0 + x1) / 2
    ax.plot([x0, x1], [cy, cy], transform=ax.transAxes, color='k', linestyle='--', linewidth=0.5)
    near, far = 1.9 * my, 4.9 * my
    for word, style, average, word_y, average_y in (('Wrong', WRONG_MARK, averages[0], cy + far, cy + near),
                                                    ('Correct', CORRECT_MARK, averages[1], cy - near, cy - far)):
        average.set_position((xc, average_y))
        label = ax.text(0, word_y, word, ha='left', **kw)
        marker_w, gap = 1.6 * mx, 0.8 * mx
        left = xc - (marker_w + gap + width(label)) / 2
        ax.add_line(_marker(left + marker_w / 2, word_y, ax.transAxes, style))
        label.set_x(left + marker_w + gap)
    return summary


def spectrum_scaling(fig, rect, eigenvalues_by_qubits, symmetrize=True):
    """3D histogram of the spectrum of H_S versus the number of qubits (Supplementary Fig. 7)."""
    ax = fig.add_axes(rect, projection='3d')
    qubits = sorted(eigenvalues_by_qubits)
    bins = np.linspace(0.0, 1.0, 22)
    hists = []
    for n in qubits:
        e = eigenvalues_by_qubits[n]
        e = symmetrize_spectrum(e) if symmetrize else np.sort(e)
        hists.append(np.histogram(e, bins=bins, density=True)[0])
    hists = np.array(hists)
    N = len(qubits)
    dy, dx = bins[1] - bins[0], 0.1
    cmap = plt.get_cmap('rainbow')
    for i, n in enumerate(qubits):
        ys = bins[:-1]
        xs = np.full_like(ys, n - dx / 2)
        ax.bar3d(xs, ys, np.zeros_like(ys), np.full_like(xs, dx), np.full_like(xs, dy), hists[i],
                 shade=True, edgecolor='k', linewidth=0.1, alpha=0.45 + i / 50,
                 color=cmap((N - 1 - i) / max(N - 1, 1)))
    ax.set_xlabel('Number of qubits', labelpad=-3)
    ax.set_ylabel('Energy', labelpad=-3)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('Frequency density', labelpad=0, rotation=90)
    ax.set_xticks(qubits)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_zlim(0, hists.max())
    for axis in ('x', 'y', 'z'):
        ax.tick_params(axis=axis, pad=-2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_linewidth(0.5)
        axis._axinfo['grid']['linewidth'] = 0.3
        axis._axinfo['tick']['inward_factor'] = 0.0
        axis._axinfo['tick']['outward_factor'] = 0.2
    ax.view_init(elev=40, azim=145)
    return ax
