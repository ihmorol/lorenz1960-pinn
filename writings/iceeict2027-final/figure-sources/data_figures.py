"""Rebuild the two numerical paper figures from the adjacent verified arrays.

Run: python figure-sources/data_figures.py
Requires numpy and matplotlib. The data manifest records the archived checkpoint
hashes, evaluation grid and agreement with the published RMSEs. No training runs.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullFormatter

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'figures'
WIDTH = 251.06 / 72
COLORS = ['#0072B2', '#D55E00', '#009E73']
NAMES = ['x', 'y', 'z']


def load(name):
    manifest = json.loads((HERE / 'data-manifest.json').read_text())
    path = HERE / (name + '.npz')
    record = manifest['runs'][name]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['data_sha256']
    data = np.load(path)
    t, pred, ref = data['t'], data['pred'], data['ref']
    assert t.shape == (13265,) and pred.shape == ref.shape == (13265, 3)
    assert np.isfinite(pred).all() and np.isfinite(ref).all()
    assert np.all(np.diff(t) > 0)
    rmse = float(np.sqrt(np.mean(np.sum((pred - ref) ** 2, axis=1))))
    assert np.isclose(rmse, record['archived_rmse_combined_l2'], rtol=2e-7)
    return t, pred, ref


def style():
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 8.5,
        'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'legend.fontsize': 8.5, 'legend.frameon': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'path.simplify': False, 'axes.linewidth': .5,
        'xtick.major.width': .5, 'ytick.major.width': .5,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.unicode_minus': True,
    })


def tidy(ax, time=True):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, color='0.88', lw=.4)
    ax.set_axisbelow(True)
    ax.tick_params(pad=2)
    ax.yaxis.set_major_locator(MaxNLocator(3))
    if time:
        ax.set_xlim(0, 13.26446)
        ax.set_xticks([0, 4, 8, 12])


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            assert text.get_fontsize() >= 8.5
            box = text.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1, (name, text.get_text(), box)
    fig.savefig(OUT / (name + '.pdf'), metadata={'Title': name, 'Author': '', 'Creator': 'Matplotlib'}, dpi=300)
    fig.savefig(OUT / (name + '.png'), dpi=300)
    plt.close(fig)


def plateau():
    t, pred, ref = load('full_interval')
    loss_path = HERE / 'full_interval_loss.csv'
    manifest = json.loads((HERE / 'data-manifest.json').read_text())
    assert hashlib.sha256(loss_path.read_bytes()).hexdigest() == manifest['runs']['full_interval']['loss_sha256']
    loss = np.genfromtxt(loss_path, delimiter=',', names=True)
    assert len(loss) == 46251 and np.all(loss['loss'] > 0)
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 3.3))
    fig.subplots_adjust(left=.15, right=.98, top=.82, bottom=.14, hspace=.65)
    for j, color in enumerate(COLORS):
        axes[0].plot(t, ref[:, j], color=color, lw=1.1, ls=(0, (4, 2)))
        axes[0].plot(t, pred[:, j], color=color, lw=.9)
        axes[1].plot(t, pred[:, j] - ref[:, j], color=color, lw=.9)
    axes[0].set_ylabel('State')
    axes[1].set_ylabel('Signed error')
    axes[1].axhline(0, color='0.4', lw=.5)
    axes[1].set_xlabel('Time t', labelpad=1)
    axes[2].semilogy(loss['iteration'], loss['loss'], color='#0072B2', lw=.65)
    axes[2].set_xlabel('Recorded optimizer evaluation', labelpad=2)
    axes[2].set_ylabel('Residual loss')
    tidy(axes[0]); tidy(axes[1]); tidy(axes[2], time=False)
    axes[0].tick_params(labelbottom=False)
    axes[2].set_yticks([1e-4, 1e-1, 1e2], labels=['1e−4', '0.1', '100'])
    axes[2].yaxis.set_minor_formatter(NullFormatter())
    axes[2].set_xticks([0, 20000, 40000])
    axes[2].set_xlim(0, loss['iteration'][-1])
    handles = [Line2D([], [], color=c, lw=1.2, label=s) for c, s in zip(COLORS, NAMES)]
    handles += [Line2D([], [], color='0.2', lw=1.1, label='PINN'), Line2D([], [], color='0.2', lw=1.1, ls=(0, (4, 2)), label='DOP853')]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(.54, 1.0), columnspacing=1.3, handlelength=2.2, labelspacing=.2)
    save(fig, 'plateau-results')


def solution():
    t, pred, ref = load('windowed')
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 3.55), sharex=True)
    fig.subplots_adjust(left=.13, right=.98, top=.83, bottom=.14, hspace=.48, wspace=.62)
    for j, (color, name) in enumerate(zip(COLORS, NAMES)):
        a, b = axes[j]
        a.plot(t, ref[:, j], color='0.55', lw=2.4, ls=(0, (4, 2)))
        a.plot(t, pred[:, j], color=color, lw=.9)
        a.set_ylabel(name + '(t)', labelpad=2)
        b.plot(t, (pred[:, j] - ref[:, j]) * 1e4, color=color, lw=.9)
        b.axhline(0, color='0.4', lw=.5)
        b.set_ylabel(name + ' error', labelpad=2)
        tidy(a); tidy(b)
    axes[0, 0].set_title('Solution', pad=5)
    axes[0, 1].set_title('Signed error (×10⁻⁴)', pad=5)
    axes[2, 0].set_xlabel('Time t', labelpad=2)
    axes[2, 1].set_xlabel('Time t', labelpad=2)
    handles = [Line2D([], [], color='0.25', lw=1, label='PINN'), Line2D([], [], color='0.55', lw=2.4, ls=(0, (4, 2)), label='DOP853')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, 1.0), ncol=2, handlelength=2.5)
    save(fig, 'solution_vs_reference')


if __name__ == '__main__':
    style()
    plateau()
    solution()
    print('Verified 13,265 points per curve; PDFs use 251.06 pt width and >=8.5 pt Times New Roman.')
