"""Used to give a general overview of the two compartment model"""
import math
from typing import Dict, List

import matplotlib
import numpy
from matplotlib.colorbar import Colorbar

from stem_cell_model import sweeper, tools
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

STEPS_ALONG_AXIS = 40
COLOR_MAP = "gnuplot"


def plot_alpha_m_against_alpha_n_and_phi_at_1():
    """Builds an image."""
    image = numpy.full((STEPS_ALONG_AXIS, STEPS_ALONG_AXIS), numpy.nan, dtype=numpy.float64)
    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_fixed_D"):
        if params.phi[0] == 1:
            image_x = int(-params.alpha[1] * (STEPS_ALONG_AXIS - 1))
            image_y = int(params.alpha[0] * (STEPS_ALONG_AXIS - 1))

            statistics = tools.get_single_parameter_set_statistics(multi_run_stats)
            image[image_y, image_x] = statistics.d_coeff_var

    return image


def plot_phi_against_opposite_alpha():
    """Builds an image."""
    image = numpy.full((STEPS_ALONG_AXIS, STEPS_ALONG_AXIS), numpy.nan, dtype=numpy.float64)
    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_fixed_D"):
        if params.alpha[0] == -params.alpha[1]:
            image_y = int(params.phi[0] * (STEPS_ALONG_AXIS - 1))
            image_x = int(params.alpha[0] * (STEPS_ALONG_AXIS - 1))
            statistics = tools.get_single_parameter_set_statistics(multi_run_stats)
            image[image_y, image_x] = statistics.d_coeff_var

    return image


def plot_along_matching_alpha_and_phi_1():
    x_values = list()
    y_values = list()
    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_fixed_D"):
        if params.alpha[0] == -params.alpha[1] :
            image_y = int(params.phi[0] * (STEPS_ALONG_AXIS - 1))
            image_x = int(params.alpha[0] * (STEPS_ALONG_AXIS - 1))
            statistics = tools.get_single_parameter_set_statistics(multi_run_stats)
            image[image_y, image_x] = statistics.d_coeff_var

    return image


fig, (ax_a, ax_b, ax_c) = plt.subplots(nrows=3,figsize=(4.181, 6.498),
                  gridspec_kw={"height_ratios": [0.47, 0.47, 0.06]})

image_phi1 = plot_alpha_m_against_alpha_n_and_phi_at_1()
image_varphi = plot_phi_against_opposite_alpha()
images_max = max(numpy.nanmax(image_varphi), numpy.nanmax(image_phi1))  # To keep the color scale of both images the same
images_max = math.ceil(images_max)


# Draw the image for phi =1
ax_a.set_title("Coeff of var $D(t)$ at $\phi = 1$")
ax_a.set_facecolor("#b2bec3")
ax_a.imshow(image_phi1, extent=(0, -1, 1, 0), aspect=1, cmap=COLOR_MAP, interpolation="nearest", vmin=0, vmax=images_max)
ax_a.invert_xaxis()
ax_a.set_xlabel("$\\alpha_m$")
ax_a.set_ylabel("$\\alpha_n$")

# Draw the image
ax_b.set_title("Coeff of var $D(t)$")
ax_b.set_facecolor("#b2bec3")
ax_b_image = ax_b.imshow(image_varphi, extent=(0, 1, 1, 0), aspect=1, cmap=COLOR_MAP, interpolation="nearest", vmin=0, vmax=images_max)
ax_b.invert_yaxis()
ax_b.set_xlabel("$\\alpha_n, -\\alpha_m$")
ax_b.set_ylabel("$\phi$")

colorbar: Colorbar = fig.colorbar(ax_b_image, cax=ax_c, orientation="horizontal")
fig.tight_layout()
plt.show()

# Now draw examples
fig, (ax_a, ax_b, ax_c) = plt.subplots(nrows=3,figsize=(4.181, 6.498))
plt.show()
