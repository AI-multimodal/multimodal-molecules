import matplotlib.pyplot as plt
import matplotlib as mpl


def set_defaults(labelsize=12, dpi=250):
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.family"] = "STIXGeneral"
    mpl.rcParams["text.usetex"] = True
    plt.rc("xtick", labelsize=labelsize)
    plt.rc("ytick", labelsize=labelsize)
    plt.rc("axes", labelsize=labelsize)
    mpl.rcParams["figure.dpi"] = dpi


def set_grids(
    ax,
    minorticks=True,
    grid=False,
    bottom=True,
    left=True,
    right=True,
    top=True,
):

    if minorticks:
        ax.minorticks_on()

    ax.tick_params(
        which="both",
        direction="in",
        bottom=bottom,
        left=left,
        top=top,
        right=right,
    )

    if grid:
        ax.grid(which="minor", alpha=0.2, linestyle=":")
        ax.grid(which="major", alpha=0.5)
