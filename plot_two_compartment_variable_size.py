import math
from typing import Dict, List, Callable, Optional

import numpy
from matplotlib.colorbar import Colorbar
from numpy import ndarray

from stem_cell_model import sweeper, tools
import matplotlib.pyplot as plt

from stem_cell_model.tools import SingleParameterSetStatistics

STEPS_ALONG_ALPHA_AND_PHI_AXIS = 40
STEPS_ALONG_SIZE_AXIS = 60
IMAGE_HEIGHT = 100
COLOR_MAP = "gnuplot"


def _get_S_to_image_y_dict(min_log_S: float, max_log_S: float) -> Dict[int, List[int]]:
    """Returns a dictionary S -> [image_y1, image_y2] so that S values can be plotted on a log scale."""
    S_to_image_y = dict()
    image_height = IMAGE_HEIGHT

    old_image_y = -1
    for log_S in numpy.linspace(min_log_S, max_log_S, num=STEPS_ALONG_SIZE_AXIS, endpoint=True):
        new_image_y_fraction =(log_S - min_log_S) / (max_log_S - min_log_S)
        new_image_y = int(new_image_y_fraction * (image_height - 1))
        S = int(10 ** log_S)

        if old_image_y == new_image_y:
            old_image_y -= 1  # Overwrite previous row

        for image_y in range(old_image_y + 1, new_image_y + 1):
            if S in S_to_image_y:
                S_to_image_y[S].append(image_y)
            else:
                S_to_image_y[S] = [image_y]
        old_image_y = new_image_y
    return S_to_image_y


def plot_S_against_opposite_alpha_and_phi_at_1(min_log_S: float, max_log_S: float, *,
                                               get_statistic: Callable[[SingleParameterSetStatistics], float]):
    # Make S to y dict to support logarithmic plotting
    S_to_image_y = _get_S_to_image_y_dict(min_log_S, max_log_S)

    # Fill the image
    image = numpy.full((IMAGE_HEIGHT, STEPS_ALONG_ALPHA_AND_PHI_AXIS), numpy.nan, dtype=numpy.float64)
    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_variable_S"):
        if params.phi[0] == 1 and params.alpha[0] == -params.alpha[1]:
            image_ys = S_to_image_y[params.S]
            image_x = int(params.alpha[0] * (STEPS_ALONG_ALPHA_AND_PHI_AXIS - 1))
            statistics = tools.get_single_parameter_set_statistics(multi_run_stats)
            for image_y in image_ys:
                image[image_y, image_x] = get_statistic(statistics)

    return image



def plot_S_against_opposite_alpha_along_phi_diagonal(min_log_S: float, max_log_S: float, *,
                                                     get_statistic: Callable[[SingleParameterSetStatistics], float]):
    # Make S to y dict to support logarithmic plotting
    S_to_image_y = _get_S_to_image_y_dict(min_log_S, max_log_S)

    # Fill the image
    image = numpy.full((IMAGE_HEIGHT, STEPS_ALONG_ALPHA_AND_PHI_AXIS), numpy.nan, dtype=numpy.float64)
    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_variable_S"):
        if params.alpha[0] == params.phi[0] and params.alpha[0] == -params.alpha[1]:
            image_ys = S_to_image_y[params.S]
            image_x = int(params.alpha[0] * (STEPS_ALONG_ALPHA_AND_PHI_AXIS - 1))
            statistics = tools.get_single_parameter_set_statistics(multi_run_stats)
            for image_y in image_ys:
                image[image_y, image_x] = get_statistic(statistics)

    return image


def main():
    min_log_S = 0
    max_log_S = 3

    def d_coeff_var(stats: SingleParameterSetStatistics):
        return stats.d_coeff_var

    def f_collapse_t(stats: SingleParameterSetStatistics):
        return stats.f_collapse

    # Plot d_coeff_var
    image_phi1 = plot_S_against_opposite_alpha_and_phi_at_1(min_log_S, max_log_S, get_statistic=d_coeff_var)
    image_varphi = plot_S_against_opposite_alpha_along_phi_diagonal(min_log_S, max_log_S, get_statistic=d_coeff_var)
    plot(name="Coeff of var $D(t)$", image_phi1=image_phi1, image_varphi=image_varphi, min_log_S=min_log_S, max_log_S=max_log_S)

    # Plot f_collapse_t
    image_phi1 = plot_S_against_opposite_alpha_and_phi_at_1(min_log_S, max_log_S, get_statistic=f_collapse_t)
    image_varphi = plot_S_against_opposite_alpha_along_phi_diagonal(min_log_S, max_log_S, get_statistic=f_collapse_t)
    plot(name="Depletion rate", image_phi1=image_phi1, image_varphi=image_varphi, min_log_S=min_log_S, max_log_S=max_log_S, vmax=10)


def plot(*, name: str, image_phi1: ndarray, image_varphi: ndarray, min_log_S: int, max_log_S: int, vmax: Optional[float] = None):
    fig, (ax_a, ax_b, ax_c) = plt.subplots(nrows=3,figsize=(4.181, 6.498),
                      gridspec_kw={"height_ratios": [0.47, 0.47, 0.06]})

    if vmax is None:
        # Calculate vmax from the images
        vmax = max(numpy.nanmax(image_varphi), numpy.nanmax(image_phi1))
        vmax = math.ceil(vmax * 10) / 10

    # Draw the image
    ax_a.set_title(name)
    ax_a.set_facecolor("#b2bec3")
    ax_a_image = ax_a.imshow(image_varphi, extent=(0, 1, max_log_S, min_log_S), aspect="auto", cmap=COLOR_MAP, interpolation="nearest", vmin=0, vmax=vmax)
    ax_a.set_xlabel("$\\phi, \\alpha_n, -\\alpha_m$")
    ax_a.set_ylabel("S")
    ax_a.set_yticks(numpy.linspace(min_log_S, max_log_S, 4))
    ax_a.set_yticklabels([f"{10 ** log_S:.0f}" for log_S in numpy.linspace(min_log_S, max_log_S, 4)])

    # Draw the image for phi =1
    ax_b.set_title(name + " at $\\phi = 1$")
    ax_b.set_facecolor("#b2bec3")
    ax_b.imshow(image_phi1, extent=(0, 1, max_log_S, min_log_S), aspect="auto", cmap=COLOR_MAP, interpolation="nearest", vmin=0, vmax=vmax)
    ax_b.set_xlabel("$\\alpha_n, -\\alpha_m$")
    ax_b.set_ylabel("S")
    ax_b.set_yticks(numpy.linspace(min_log_S, max_log_S, 4))
    ax_b.set_yticklabels([f"{10 ** log_S:.0f}" for log_S in numpy.linspace(min_log_S, max_log_S, 4)])

    colorbar: Colorbar = fig.colorbar(ax_a_image, cax=ax_c, orientation="horizontal")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
