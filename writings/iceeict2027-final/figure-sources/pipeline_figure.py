from pathlib import Path

W = 816
OY = 0
nodes, wires = [], []

CSS = """
.pipe { position: relative; width: 816pt; color: #334155; line-height: 1; font-family: "Times New Roman", Times, serif; }
.pipe .wire { position: absolute; z-index: 1; left: 0; top: 0; width: 100%; height: 100%; overflow: visible; }
.pipe .wire text { font-family: "Times New Roman", Times, serif; font-size: 27.75px; font-weight: 700; }
.pipe .n { position: absolute; display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 2pt;
  text-align: center; font-size: 27.75pt; border: 2pt solid; border-radius: 10pt; padding: 0 14pt; }
.pipe .n.bare { z-index: 2; border: 0; padding: 0; }
.pipe .n b { font-size: 31pt; color: #0f172a; }
.pipe .n.mini { z-index: 2; font-size: 27.75pt; padding: 0 8pt; }
.pipe .n.mini b { font-size: 29.5pt; }
.pipe .n.dia b { font-size: 30pt; }
.pipe .n.dia { font-size: 27.75pt; }
.pipe .num { position: absolute; left: 10pt; top: 10pt; width: 34pt; height: 34pt; border-radius: 50%; color: #fff;
  font-size: 27.75pt; font-weight: 700; line-height: 34pt; text-align: center; }
.pipe .frame { position: absolute; border: 2pt dashed #065f46; border-radius: 14pt; background: #fff; }
.pipe .tag { position: absolute; z-index: 2; height: 38pt; line-height: 38pt; padding: 0 14pt; border-radius: 19pt; background: #065f46; color: #fff;
  font-size: 27.75pt; font-weight: 700; letter-spacing: .02em; }
.pipe mjx-container { margin: 0 !important; font-size: .98em; }
"""


def node(x, y, w, h, title, lines=(), fill="#fff", stroke="#334155", num=None, cls=""):
    body = (f"<b>{title}</b>" if title else "") + "".join(f"<div>{ln}</div>" for ln in lines)
    badge = f'<span class="num" style="background:{stroke}">{num}</span>' if num else ""
    nodes.append(f'<div class="n {cls}" style="left:{x}pt;top:{y + OY}pt;width:{w}pt;height:{h}pt;'
                 f'background:{fill};border-color:{stroke}">{badge}{body}</div>')
    return y + h


def box(x, y, w, h, title, lines=(), fill="#fff", stroke="#334155", num=None):
    return node(x, y, w, h, title, lines, fill, stroke, num)


def diamond(cx, cy, hw, hh, title, lines=()):
    c = cy + OY
    wires.append(f'<polygon points="{cx},{c - hh} {cx + hw},{c} {cx},{c + hh} {cx - hw},{c}" fill="#fefce8" stroke="#a16207" stroke-width="2"/>')
    node(cx - hw * .6, cy - hh, hw * 1.2, 2 * hh, title, lines, "transparent", "transparent", cls="bare dia")


MK = {"#475569": "pa", "#c2410c": "pe", "#065f46": "pw"}


def arrow(pts, color="#475569", dash=False, label=None, lx=0, ly=0, rot=False):
    points = " ".join(f"{x},{y + OY}" for x, y in pts)
    ly += OY
    dashed = ' stroke-dasharray="9 6"' if dash else ""
    wires.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"{dashed} stroke-linejoin="round" marker-end="url(#{MK[color]})"/>')
    if label:
        transform = f' transform="rotate(-90 {lx} {ly})"' if rot else ""
        wires.append(f'<text x="{lx}" y="{ly}" fill="{color}" text-anchor="middle"{transform}>{label}</text>')


def down(x, y0, y1):
    arrow([(x, y0), (x, y1 - 2)])


L, R, MID, LC = 40, 776, 408, 230

y = box(4, 4, 808, 148, "Problem setup", [
    r"$\dot x=-0.1\,yz,\quad \dot y=1.6\,xz,\quad \dot z=-0.75\,xy$",
    r"$u_0=(0.5,\,0.75,\,1.0),\qquad t\in[0,\,13.26446]$",
    "27 windows · 1,536 points/window · float64"], "#fff7ed", "#c2410c", 1)
frame_top = 185
down(MID, y, 224)
OY = 48

y2 = 176
y = node(L, y2, R - L, 180, "", (), "#eff6ff", "#1e40af", 2)
nodes.append(f'<div class="n bare" style="left:{L}pt;top:{y2 + 14 + OY}pt;width:{R - L}pt;height:38pt"><b>PINN for window $k$</b></div>')
x = 58
subblocks = [
    (92, "Input", ["$t$"]),
    (132, "Hidden", ["4 × 60", "tanh"]),
    (140, "Output", ["$N_k(t)$", r"$\in\mathbb{R}^3$"]),
    (264, "Hard IC", [r"$\hat u_k=u_{0,k}$", r"$+(t-t_k)\,N_k(t)$"]),
]
for i, (w, title, lines) in enumerate(subblocks):
    node(x, y2 + 66, w, 98, title, lines, "#fff", "#93c5fd", cls="mini")
    if i < 3:
        arrow([(x + w + 3, y2 + 113), (x + w + 23, y2 + 113)])
    x += w + 26
mid2 = y2 + 90

down(MID, y, 384)
y = box(L, 384, R - L, 100, "ODE residual", [
    r"$r_k=d\hat u_k/dt-f(\hat u_k)$; derivatives by autodiff"], "#f5f3ff", "#6d28d9", 3)
down(MID, y, 514)
y4 = 514
y = box(L, y4, R - L, 152, "Causal residual loss", [
    r"$\mathcal L_i=\mathrm{mean}(r_i^2),\quad w_i=\exp\left(-\varepsilon\textstyle\sum_{j\lt i}\mathcal L_j\right)$",
    r"$\mathcal L=\mathrm{mean}(w_i\mathcal L_i)$; detach weights for gradients",
    r"$\varepsilon:\ 0.01\to 0.1\to 1\to 10$"], "#f5f3ff", "#6d28d9", 4)
mid4 = y4 + 74
down(MID, y, 692)
y = box(L, 692, R - L, 94, "Adam optimization", [
    "learning rate × 0.9 every 1,000 steps"], "#f0fdfa", "#0d9488", 5)
arrow([(LC, y), (LC, 806)])

cy, hh = 918, 110
diamond(LC, cy, 190, hh, "Stage complete?", [r"$\min w>0.99$", "or 4,000 steps"])
diamond(615, cy, 145, hh, r"All $\varepsilon$ values used?")
arrow([(420, cy), (468, cy)], label="YES", lx=444, ly=cy - 13)
arrow([(40, cy), (22, cy), (22, mid4), (38, mid4)], label="NO", lx=47, ly=cy + 34)
row = 1062
arrow([(615, cy + hh), (615, row - 2)], label="NO", lx=650, ly=row - 10)
arrow([(542, cy + hh / 2), (542, row - 22), (LC, row - 22), (LC, row - 2)], label="YES", lx=390, ly=row - 27)
box(480, row, 270, 100, r"Next $\varepsilon$", ["keep parameters"], "#fff7ed", "#c2410c")
arrow([(750, row + 50), (792, row + 50), (792, mid4), (778, mid4)], "#c2410c", True, "next ε", 787, 840, True)

y = box(L, row, 380, 118, "L-BFGS", ["≤ 1,000 iterations", "unweighted residual MSE"], "#f0fdfa", "#0d9488", 6)
down(LC, y, 1210)
y = box(L, 1210, 380, 98, "Save window", ["checkpoint · history · prediction"], "#f8fafc", "#475569", 7)
arrow([(LC, y), (LC, 1318)])

cy, hh = 1390, 70
diamond(LC, cy, 180, hh, "Last window?", ["$k=26$"])
box(470, 1328, 306, 130, "Next window", [
    r"$u_{0,k+1}=\hat u_k(t_{k+1})$", "copy parameters", r"$k\leftarrow k+1$"], "#ecfdf5", "#065f46")
arrow([(410, cy), (468, cy)], label="NO", lx=439, ly=cy - 13)
arrow([(776, cy), (806, cy), (806, mid2), (778, mid2)], "#065f46", True, "next window", 800, (mid2 + mid4) / 2, True)
frame_bot = 1478 + OY
arrow([(LC, cy + hh), (LC, 1515)], label="YES", lx=274, ly=1504)

y = box(4, 1517, 808, 120, "Reference evaluation", [
    r"DOP853 · 13,265 points · rtol $10^{-10}$ · atol $10^{-12}$",
    "evaluation only; not used for optimization"], "#eff6ff", "#1e40af", 8)
down(MID, y, 1667)
y = box(4, 1667, 808, 96, "Evaluation outputs", [
    "MAE · RMSE · maximum error · residual MSE",
    "invariant drift · training cost"], "#f8fafc", "#475569", 9)
H = y + 4 + OY

markers = "".join(
    f'<marker id="{marker}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>'
    for color, marker in MK.items())
frame = (f'<div class="frame" style="left:4pt;top:{frame_top}pt;width:808pt;height:{frame_bot - frame_top}pt"></div>'
         f'<div class="tag" style="left:24pt;top:{frame_top - 19}pt">REPEAT FOR EACH WINDOW <i>k</i> = 0 … 26</div>')
pipe = (f'<style>{CSS}</style><div class="pipe" style="height:{H}pt">{frame}'
        f'<svg class="wire" viewBox="0 0 {W} {H}"><defs>{markers}</defs>{"".join(wires)}</svg>{"".join(nodes)}</div>')

source = Path(__file__).with_name("pipeline.html")
source.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Sequential windowed PINN training</title>
<!-- Derived from original-pipeline.html; layout preserved, typography enlarged for print. -->
<script>window.MathJax={{tex:{{inlineMath:[["$","$"]]}},svg:{{fontCache:"global"}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>@page {{ size: {W}pt {H}pt; margin: 0; }} * {{ box-sizing: border-box; margin: 0; }}
html, body {{ width: {W}pt; height: {H}pt; overflow: hidden; }}
body {{ font-family: "Times New Roman", Times, serif; background: #fff; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}</style>
</head><body>{pipe}</body></html>""", encoding="utf-8")
print(f"wrote {source} ({W} x {H} pt)")
