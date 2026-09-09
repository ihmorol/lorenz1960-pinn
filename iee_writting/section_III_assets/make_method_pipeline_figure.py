"""Create the Section III PINN-method pipeline figure from the locked run facts."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT = Path(__file__).resolve().parent / "figures" / "fig01_pinn_method_pipeline"


def box(ax, xy, width, height, title, body, face, edge="#25364a"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.2,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(patch)
    x, y = xy
    ax.text(x + width / 2, y + height * 0.66, title, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#17212b")
    ax.text(x + width / 2, y + height * 0.34, body, ha="center", va="center",
            fontsize=7.7, linespacing=1.25, color="#17212b")


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": "#34495e"})


def main():
    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=120)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.02, 0.40), 0.13, 0.22, "Input", "$t\\in[0,1]$", "#e8f1f8")
    box(ax, (0.19, 0.34), 0.19, 0.34, "MLP $\\mathcal{N}(t;\\theta)$",
        "$1\\to60\\to60$\n$60\\to60\\to3$\n$\\tanh$; Xavier init",
        "#edf5e8")
    box(ax, (0.42, 0.34), 0.20, 0.34, "Hard trial solution",
        "$u_T=u_0+g(t)\\mathcal{N}(t;\\theta)$\n$g(t)=(t-t_0)/(t_f-t_0)$\n$u_T(t_0)=u_0$",
        "#fff3df")
    box(ax, (0.66, 0.57), 0.15, 0.22, "Autograd", "$du_T/dt$", "#f3e9f6")
    box(ax, (0.66, 0.20), 0.15, 0.22, "ODE right side",
        "quadratic products\nof $(x_T,y_T,z_T)$", "#f3e9f6")
    box(ax, (0.86, 0.35), 0.12, 0.30, "Residual + loss",
        "$r=du_T/dt-f(u_T)$\n$\\mathcal{L}=\\mathrm{mean}(r^2)$\n$N_c=3000$ LHS",
        "#fbe8e8")

    arrow(ax, 0.15, 0.51, 0.19, 0.51)
    arrow(ax, 0.38, 0.51, 0.42, 0.51)
    arrow(ax, 0.62, 0.57, 0.66, 0.68)
    arrow(ax, 0.62, 0.45, 0.66, 0.31)
    arrow(ax, 0.81, 0.68, 0.86, 0.57)
    arrow(ax, 0.81, 0.31, 0.86, 0.43)

    ax.text(0.50, 0.90, "Adam: 20,000 iterations | learning rate $10^{-3}\\to10^{-4}$",
            ha="center", va="center", fontsize=8.0, color="#17212b")
    ax.text(0.50, 0.06, "DOP853 reference: evaluation only | separate 1,001-point grid",
            ha="center", va="center", fontsize=7.4, color="#4a5560")

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=180)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
