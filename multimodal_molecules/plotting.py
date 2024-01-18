import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.interpolate import interpn


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


# https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
def density_scatter(x, y, ax, sort=True, bins=20, **kwargs):
    """
    Scatter plot colored by 2d histogram
    """

    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([x, y]).T,
        method="splinef2d",
        bounds_error=False,
    )

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    scat = ax.scatter(x, y, c=z, **kwargs)
    return scat


def remove_axis_spines(ax, visible=True):
    # ax.spines["right"].set_visible(False)
    # ax.spines["top"].set_visible(False)
    for pos in ["left", "right", "top", "bottom"]:
        ax.spines[pos].set_linewidth(0.5)
        if not visible:
            ax.spines[pos].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
