"""Render the concise training diagram at its final single-column print size.

Derived from pipeline_figure_original.py and the manuscript algorithm. Scientific
values and training stages are unchanged; detailed equations remain in the text.
"""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
WIDTH, HEIGHT = 251.06, 225.0
FONT = 'Times New Roman'
FONT_FILE = Path(font_manager.findfont(FONT, fallback_to_default=False))
plt.rcParams.update({
    'font.family': FONT, 'font.size': 8.5, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'svg.fonttype': 'none',
    'mathtext.fontset': 'custom', 'mathtext.rm': FONT,
    'mathtext.it': FONT + ':italic', 'mathtext.bf': FONT + ':bold',
    'mathtext.fallback': None,
})
fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
ax.axis('off')
X, BOX_W = 5.0, 228.0
CENTER = X + BOX_W / 2
texts = []
boxes = []

def box(y, h, title, lines, fill, edge, title_size=9.3):
    patch = FancyBboxPatch((X, y), BOX_W, h,
                          boxstyle='round,pad=0,rounding_size=3',
                          linewidth=0.7, edgecolor=edge, facecolor=fill)
    ax.add_patch(patch)
    rows = [(title, title_size, 'bold')] + [(s, 8.5, 'normal') for s in lines]
    line_height = 10
    first_y = y + (h - line_height * (len(rows) - 1)) / 2
    for i, (label, size, weight) in enumerate(rows):
        text = ax.text(CENTER, first_y + i * line_height, label,
                       ha='center', va='center', fontsize=size,
                       fontweight=weight, color='#17202a')
        texts.append((text, patch))
    boxes.append((y, h))

box(2, 32, 'Lorenz-1960 initial-value problem', [
    'u(0) = (0.5, 0.75, 1.0),  T = 13.26446',
    '27 equal time windows; float64 precision'], '#fff7ed', '#9a3412')
box(40, 34, 'Separate PINN for each window k = 0, …, 26', [
    '4 × 60 tanh; 1,536 collocation points',
    'Hard trial solution enforces the initial state'],
    '#eff6ff', '#1e40af', 8.9)
box(80, 34, 'Causal Adam: ε = 0.01, 0.1, 1, 10', [
    'Recompute causal weights; detach for gradients',
    'Advance at verified min w > 0.99 or 4,000 updates/stage'],
    '#f5f3ff', '#6d28d9')
box(120, 24, 'L-BFGS: unweighted residual mean square', [
    'At most 1,000 iterations per window'], '#f0fdfa', '#0f766e', 8.9)
box(150, 34, 'Save the window; hand off if k < 26', [
    'Next initial state = predicted endpoint',
    'Copy trained parameters; then train window k + 1'],
    '#ecfdf5', '#166534')
box(190, 33, 'After 27 windows: compare with DOP853', [
    '13,265 times; MAE, RMSE and maximum error',
    'Residual MSE, invariant drift and training cost'],
    '#f8fafc', '#475569', 8.9)
for (y, h), (next_y, _) in zip(boxes[:-1], boxes[1:]):
    ax.add_patch(FancyArrowPatch((CENTER, y + h + 0.3),
                                (CENTER, next_y - 0.3),
                                arrowstyle='-|>', mutation_scale=6,
                                linewidth=0.75, color='#475569'))
# The right-hand return path applies only to the conditional handoff box.
ax.plot([233, 242, 242, 233], [167, 167, 57, 57],
        color='#166534', linewidth=0.75)
ax.add_patch(FancyArrowPatch((242, 57), (233.1, 57), arrowstyle='-|>',
                            mutation_scale=6, linewidth=0.75, color='#166534'))
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
# Text must remain inside its box at the actual print size.
for text, patch in texts:
    bb, pb = text.get_window_extent(renderer), patch.get_window_extent(renderer)
    assert bb.x0 >= pb.x0 + 2 and bb.x1 <= pb.x1 - 2, text.get_text()
    assert bb.y0 >= pb.y0 + 1 and bb.y1 <= pb.y1 - 1, text.get_text()
    assert text.get_fontsize() >= 8.5
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / 'method.pdf', metadata={
    'Title': 'Sequential windowed PINN training', 'Author': '',
    'Subject': 'Protocol diagram', 'Keywords': ''})
fig.savefig(OUT / 'method.png', dpi=300)
fig.savefig(Path(__file__).with_name('pipeline.svg'))
plt.close(fig)
import fitz
pdf = fitz.open(OUT / 'method.pdf')
sizes = [span['size'] for page in pdf for block in page.get_text('dict')['blocks']
         if 'lines' in block for line in block['lines'] for span in line['spans']]
fonts = [{
    'name': font[3], 'embedded': len(pdf.extract_font(font[0])[3]) > 0,
    'type': font[2]
} for font in pdf[0].get_fonts(full=True)]
assert min(sizes) >= 8.49, min(sizes)
assert all(font['embedded'] and font['type'] != 'Type3' for font in fonts)
assert abs(pdf[0].rect.width - WIDTH) < 0.01
metadata = {
    'source': Path(__file__).name,
    'derived_from': 'pipeline_figure_original.py and manuscript sequential training algorithm (Fig. 3)',
    'width_pt': WIDTH, 'height_pt': HEIGHT,
    'minimum_pdf_text_size_pt': min(sizes), 'fonts': fonts,
    'font_file': FONT_FILE.name,
    'font_file_sha256': hashlib.sha256(FONT_FILE.read_bytes()).hexdigest(),
    'protocol': {
        'windows': 27, 'window_indices': [0, 26],
        'network': 'four hidden layers of 60 tanh units; one network per window',
        'hard_initial_state': True, 'points_per_window': 1536,
        'causal_adam_epsilon': [0.01, 0.1, 1, 10],
        'causal_weights_detached_for_gradient': True,
        'stage_threshold': 'minimum causal weight > 0.99',
        'stage_cap_adam_updates': 4000,
        'lbfgs_objective': 'unweighted residual mean square',
        'lbfgs_iterations_per_window_at_most': 1000,
        'handoff': 'predicted endpoint and copied trained parameters',
        'reference': 'DOP853, 13265 evaluation times; not an optimization target',
    },
    'pdf_sha256': hashlib.sha256((OUT / 'method.pdf').read_bytes()).hexdigest(),
}
Path(__file__).with_name('pipeline-metadata.json').write_text(
    json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
print(json.dumps(metadata, indent=2))
