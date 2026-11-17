"""Optional plot of the E3 block-error-rate curves (requires the `plot` extra)."""

from collections.abc import Sequence
from pathlib import Path

from .experiments import BlerCurve

# Categorical slots 1–4 of the validated reference palette (light surface).
# Two of them are below 3:1 contrast with the surface, so each curve also gets
# its own marker shape and a direct label; the printed E3 table serves as the
# table view.
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
_MARKERS = ["o", "s", "D", "^"]
_SURFACE = "#fcfcfb"
_TEXT = "#1f1f1e"
_MUTED = "#6b6b66"
_GRID = "#e4e3df"


def plot_bler(curves: Sequence[BlerCurve], path: Path, title: str) -> Path:
    """Draw BLER vs p on a log scale and save it as a PNG.

    Simulated points with zero observed errors cannot be shown on a log axis,
    so they are omitted from their curve.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    simulated = [c for c in curves if not c.label.endswith("(exact)")]
    exact = [c for c in curves if c.label.endswith("(exact)")]
    for i, curve in enumerate(simulated):
        color = _SERIES[i % len(_SERIES)]
        pts = [(p, b) for p, b in zip(curve.ps, curve.bler) if b > 0]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=2, marker=_MARKERS[i % 4],
                markersize=8, markeredgecolor=_SURFACE, markeredgewidth=2,
                label=curve.label)
        # No direct end labels: every curve converges near BLER = 1 at large p,
        # so labels there collide. Identity comes from the legend plus a
        # distinct marker shape for each curve.
    for curve in exact:
        # The exact curve belongs to the same entity as the simulated
        # repetition curve, so it reuses that curve's colour, drawn dashed.
        idx = next(
            (i for i, c in enumerate(simulated) if curve.label.startswith(c.label)), 0
        )
        ax.plot(curve.ps, curve.bler, color=_SERIES[idx % len(_SERIES)], linewidth=2,
                linestyle="--", label=curve.label)

    ax.set_yscale("log")
    ax.set_xlabel("bit-flip probability p", color=_TEXT)
    ax.set_ylabel("block error rate", color=_TEXT)
    ax.set_title(title, color=_TEXT, fontsize=11, loc="left", pad=16)
    ax.text(0.0, 1.01, "simulated points with zero observed errors are not drawn",
            transform=ax.transAxes, fontsize=8, color=_MUTED, va="bottom")
    ax.grid(True, which="major", color=_GRID, linewidth=0.8)
    ax.tick_params(colors=_MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.legend(frameon=False, fontsize=9, labelcolor=_TEXT, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)
    return path
