import math
from typing import Dict, List

import numpy
from matplotlib.colorbar import Colorbar

from stem_cell_model import sweeper, tools
import matplotlib.pyplot as plt

STEPS_ALONG_AXIS = 40
IMAGE_HEIGHT = 100
COLOR_MAP = "gnuplot"


def _get_S_to_image_y_dict(min_log_S: float, max_log_S: float) -> Dict[int, List[int]]:
    """Returns a dictionary S -> [image_y1, image_y2] so that S values can be plotted on a log scale."""
    S_to_image_y = dict()
    image_height = IMAGE_HEIGHT

    old_image_y = -1
    for log_S in numpy.linspace(min_log_S, max_log_S, num=STEPS_ALONG_AXIS, endpoint=True):
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


def plot_S_against_opposite_alpha_and_phi_at_1(min_log_S: float, max_log_S: float):
    # Make S to y dict to support logarithmic plotting
    S_to_image_y = _get_S_to_image_y_dict(min_log_S, max_log_S)

    # Fill the image
    image = numpy.full((IMAGE_HEIGHT, STEPS_ALONG_AXIS), numpy.nan, dtype=numpy.float64)
    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_variable_S"):
        if params.phi[0] == 1 and params.alpha[0] == -params.alpha[1]:
            image_ys = S_to_image_y[params.S]
            image_x = int(params.alpha[0] * (STEPS_ALONG_AXIS - 1))
            statistics = tools.get_single_parameter_set_statistics(multi_run_stats)
            for image_y in image_ys:
                image[image_y, image_x] = statistics.d_coeff_var

    return image



def plot_S_against_opposite_alpha_along_phi_diagonal(min_log_S: float, max_log_S: float):
    # Make S to y dict to support logarithmic plotting
    S_to_image_y = _get_S_to_image_y_dict(min_log_S, max_log_S)

    # Fill the image
    image = numpy.full((IMAGE_HEIGHT, STEPS_ALONG_AXIS), numpy.nan, dtype=numpy.float64)
    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_variable_S"):
        if params.alpha[0] == params.phi[0] and params.alpha[0] == -params.alpha[1]:
            image_ys = S_to_image_y[params.S]
            image_x = int(params.alpha[0] * (STEPS_ALONG_AXIS - 1))
            statistics = tools.get_single_parameter_set_statistics(multi_run_stats)
            for image_y in image_ys:
                image[image_y, image_x] = statistics.d_coeff_var

    return image


fig, (ax_a, ax_b, ax_c) = plt.subplots(nrows=3,figsize=(4.181, 6.498),
                  gridspec_kw={"height_ratios": [0.47, 0.47, 0.06]})
min_log_S = 0
max_log_S = 2

image_phi1 = plot_S_against_opposite_alpha_and_phi_at_1(min_log_S, max_log_S)
image_varphi = plot_S_against_opposite_alpha_along_phi_diagonal(min_log_S, max_log_S)
images_max = max(numpy.nanmax(image_varphi), numpy.nanmax(image_phi1))  # To keep the color scale of both images the same
images_max = math.ceil(images_max * 10) / 10

# Draw the image
ax_a.set_title("Coeff of var $D(t)$")
ax_a.set_facecolor("#eeeeee")
ax_a_image = ax_a.imshow(image_varphi, extent=(0, 1, max_log_S, min_log_S), aspect="auto", cmap=COLOR_MAP, interpolation="nearest", vmin=0, vmax=images_max)
ax_a.set_xlabel("$\\phi, \\alpha_n, -\\alpha_m$")
ax_a.set_ylabel("S")
ax_a.set_yticks(numpy.linspace(min_log_S, max_log_S, 3))
ax_a.set_yticklabels([f"{10 ** log_S:.0f}" for log_S in numpy.linspace(min_log_S, max_log_S, 3)])

# Draw the image for phi =1
ax_b.set_title("Coeff of var $D(t)$ at $\\phi = 1$")
ax_b.set_facecolor("#eeeeee")
ax_b.imshow(image_phi1, extent=(0, 1, max_log_S, min_log_S), aspect="auto", cmap=COLOR_MAP, interpolation="nearest", vmin=0, vmax=images_max)
ax_b.set_xlabel("$\\alpha_n, -\\alpha_m$")
ax_b.set_ylabel("S")
ax_b.set_yticks(numpy.linspace(min_log_S, max_log_S, 3))
ax_b.set_yticklabels([f"{10 ** log_S:.0f}" for log_S in numpy.linspace(min_log_S, max_log_S, 3)])

colorbar: Colorbar = fig.colorbar(ax_a_image, cax=ax_c, orientation="horizontal")
fig.tight_layout()
plt.show()
