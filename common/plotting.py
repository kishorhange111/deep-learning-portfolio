"""
One matplotlib style for every figure in the newer projects: a fixed categorical colour order
(colour-vision-deficiency-checked for adjacent series), thin lines, recessive grid and axes, no
top/right spines. Colour follows the entity: pass the same `key` and you get the same colour in
every figure.
"""

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def apply_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "font.size": 10,
            "font.family": ["DejaVu Sans"],
        }
    )
    return plt


class ColorMap:
    """Stable entity -> colour assignment in fixed slot order (never cycled past 8)."""

    def __init__(self, keys=()):
        self._map = {}
        for k in keys:
            self.__getitem__(k)

    def __getitem__(self, key):
        if key not in self._map:
            if len(self._map) >= len(SERIES):
                raise ValueError("more than 8 series: fold extras into 'Other' or use small multiples")
            self._map[key] = SERIES[len(self._map)]
        return self._map[key]
