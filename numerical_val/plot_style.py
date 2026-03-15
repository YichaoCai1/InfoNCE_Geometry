import matplotlib.pyplot as plt


CURVE_FIGSIZE = (3.35, 2.6)
SQUARE_FIGSIZE = (3.35, 3.0)
SHORT_FIGSIZE = (3.35, 2.25)
THREE_D_PANEL_SIZE = (2.75, 2.8)

PRIMARY_COLOR = "#355f8d"
SECONDARY_COLOR = "#d68612"
ACCENT_COLOR = "#443983"
HIGHLIGHT_COLOR = "#c43c24"
TERTIARY_COLOR = "#2a9d8f"
QUATERNARY_COLOR = "#8c564b"
GRID_COLOR = "#c9d5df"
DIAGONAL_COLOR = "#667085"
HEATMAP_CMAP = "viridis"
SURFACE_CMAP = "viridis"
SERIES_COLORS = (
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    ACCENT_COLOR,
    HIGHLIGHT_COLOR,
    TERTIARY_COLOR,
    QUATERNARY_COLOR,
)

def set_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "axes.titlepad": 4,
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "axes.edgecolor": "#5b6574",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.fontsize": 8.5,
        "legend.frameon": True,
        "legend.fancybox": True,
        "legend.framealpha": 0.92,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def prettify_ax(ax):
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.85, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def styled_legend(ax, loc="best", ncol=1, **kwargs):
    legend = ax.legend(
        loc=loc,
        ncol=ncol,
        borderpad=0.35,
        handletextpad=0.45,
        columnspacing=0.8,
        **kwargs,
    )
    if legend is not None:
        legend.get_frame().set_edgecolor("0.85")
        legend.get_frame().set_linewidth(0.8)
    return legend


def build_series_color_map(labels):
    ordered_labels = list(dict.fromkeys(labels))
    if not ordered_labels:
        return {}
    return {
        label: SERIES_COLORS[idx % len(SERIES_COLORS)]
        for idx, label in enumerate(ordered_labels)
    }


def grid_figure_size(nrows, ncols, panel_size=THREE_D_PANEL_SIZE):
    return (panel_size[0] * ncols, panel_size[1] * nrows)
