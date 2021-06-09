import math
from typing import Dict, List

import numpy
from matplotlib.colorbar import Colorbar

from stem_cell_model import sweeper, tools
import matplotlib.pyplot as plt

STEPS_ALONG_PHI_AXIS = 40
STEPS_ALONG_SIZE_AXIS = 40
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


def plot_coeff_of_variation_for_S_against_phi(min_log_S: float, max_log_S: float):
    # Make S to y dict to support logarithmic plotting
    S_to_image_y = _get_S_to_image_y_dict(min_log_S, max_log_S)

    # Fill the image
    image = numpy.full((IMAGE_HEIGHT, STEPS_ALONG_PHI_AXIS), numpy.nan, dtype=numpy.float64)
    for params, multi_run_stats in sweeper.load_sweep_results("one_comp_sweep_data_var_D_var_phi"):
        image_ys = S_to_image_y[params.S]
        image_x = round(params.phi[0] * (STEPS_ALONG_PHI_AXIS - 1))
        statistics = tools.get_single_parameter_set_statistics(multi_run_stats)
        for image_y in image_ys:
            image[image_y, image_x] = statistics.d_coeff_var

    return image


def plot_depletion_for_S_against_phi(min_log_S: float, max_log_S: float):
    # Make S to y dict to support logarithmic plotting
    S_to_image_y = _get_S_to_image_y_dict(min_log_S, max_log_S)

    # Fill the image
    image = numpy.full((IMAGE_HEIGHT, STEPS_ALONG_PHI_AXIS), numpy.nan, dtype=numpy.float64)
    for params, multi_run_stats in sweeper.load_sweep_results("one_comp_sweep_data_var_D_var_phi"):
        image_ys = S_to_image_y[params.S]
        image_x = round(params.phi[0] * (STEPS_ALONG_PHI_AXIS - 1))
        statistics = tools.get_single_parameter_set_statistics(multi_run_stats)
        for image_y in image_ys:
            image[image_y, image_x] = statistics.f_collapse

    return image


fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(nrows=2,ncols=2,figsize=(6.498, 3.9),
                  gridspec_kw={"height_ratios": [0.94, 0.06]})
min_log_S = 0
max_log_S = 2

image_cov = plot_coeff_of_variation_for_S_against_phi(min_log_S, max_log_S)
image_cov_max = math.ceil(numpy.nanmax(image_cov) * 10) / 10

image_depletion = plot_depletion_for_S_against_phi(min_log_S, max_log_S)
image_depletion_max = math.ceil(numpy.nanmax(image_depletion) * 10) / 10

# Draw the coefficient of variation image
ax_a.set_title("Coefficient of variation in $D(t)$")
ax_a.set_facecolor("#b2bec3")
ax_a_image = ax_a.imshow(image_cov, extent=(0, 1, max_log_S, min_log_S), aspect="auto", cmap=COLOR_MAP, interpolation="nearest", vmin=0, vmax=image_cov_max)
ax_a.set_xlabel("$\\phi$")
ax_a.set_ylabel("$D$")
ax_a.set_yticks(numpy.linspace(min_log_S, max_log_S, 3))
ax_a.set_yticklabels([f"{10 ** log_S:.0f}" for log_S in numpy.linspace(min_log_S, max_log_S, 3)])

# Draw the depletion image
ax_b.set_title("Depletion rate / 1000h")
ax_b.set_facecolor("#b2bec3")
ax_b_image = ax_b.imshow(image_depletion, extent=(0, 1, max_log_S, min_log_S), aspect="auto", cmap=COLOR_MAP, interpolation="nearest", vmin=0, vmax=image_depletion_max)
ax_b.set_xlabel("$\\phi$")
ax_b.set_ylabel("$D$")
ax_b.set_yticks(numpy.linspace(min_log_S, max_log_S, 3))
ax_b.set_yticklabels([f"{10 ** log_S:.0f}" for log_S in numpy.linspace(min_log_S, max_log_S, 3)])

fig.colorbar(ax_a_image, cax=ax_c, orientation="horizontal")
fig.colorbar(ax_b_image, cax=ax_d, orientation="horizontal")
fig.tight_layout()
plt.show()
