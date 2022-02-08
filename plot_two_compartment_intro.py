"""Used to give a general overview of the two compartment model"""
import math
from collections import OrderedDict
from typing import Dict, List, Tuple

import matplotlib
import numpy
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar

from stem_cell_model import sweeper, tools, two_compartment_model
import matplotlib.pyplot as plt

from stem_cell_model.parameters import SimulationConfig, SimulationParameters
from stem_cell_model.results import SimulationResults

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


def get_example_line_for_phi_1_opposite_alpha() -> Tuple[List[float], List[float]]:
    """Line along the largest variation."""
    values = dict()
    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_fixed_D"):
        if params.alpha[0] == -params.alpha[1] and params.phi[0] == 1:
            statistics = tools.get_single_parameter_set_statistics(multi_run_stats)
            values[params.alpha[0]] = statistics.d_coeff_var

    x_values = list()
    y_values = list()
    for key, value in sorted(values.items()):
        x_values.append(key)
        y_values.append(value)
    return x_values, y_values


def get_example_line_for_opposite_alpha_and_matching_phi() -> Tuple[List[float], List[float]]:
    """Line along the lowest phi for every alpha. This is one of the optimal lines for minimizing fluctuations.."""
    values = dict()
    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_fixed_D"):
        if params.alpha[0] == -params.alpha[1] and params.phi[0] == params.alpha[0]:
            statistics = tools.get_single_parameter_set_statistics(multi_run_stats)
            values[params.alpha[0]] = statistics.d_coeff_var

    x_values = list()
    y_values = list()
    for key, value in sorted(values.items()):
        x_values.append(key)
        y_values.append(value)
    return x_values, y_values


def get_example_line_for_opposite_alpha_and_higher_phi() -> Tuple[List[float], List[float]]:
    """Line along (0.5 * lowest phi) + 0.5 for every alpha. So we're halfway the lowest phi
    and the maximal phi (which is always 1)."""
    values = dict()
    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_fixed_D"):
        if params.alpha[0] == -params.alpha[1] and (0.5 * params.alpha[0]) + 0.5 == params.phi[0]:
            statistics = tools.get_single_parameter_set_statistics(multi_run_stats)
            values[params.alpha[0]] = statistics.d_coeff_var

    x_values = list()
    y_values = list()
    for key, value in sorted(values.items()):
        x_values.append(key)
        y_values.append(value)
    return x_values, y_values


def get_second_example_line_for_phi_1() -> Tuple[List[float], List[float]]:
    """The line for phi=1 and a_n = -a_m + 0.5."""
    values = dict()
    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_fixed_D"):
        if params.alpha[0] == -params.alpha[1] + 0.5 and params.phi[0] == 1:
            statistics = tools.get_single_parameter_set_statistics(multi_run_stats)
            values[params.alpha[0]] = statistics.d_coeff_var

    x_values = list()
    y_values = list()
    for key, value in sorted(values.items()):
        x_values.append(key)
        y_values.append(value)
    return x_values, y_values


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

# Now draw values along lines
fig, (ax_a, ax_b) = plt.subplots(nrows=2,figsize=(4.181, 4.498), sharex="col", sharey="col")
ax_a.plot(*get_example_line_for_phi_1_opposite_alpha(), color="#81ecec", label="$\\phi=1,\\alpha_n=-\\alpha_m$")
ax_a.plot(*get_second_example_line_for_phi_1(), color="#fd79a8", label="$\\phi=1,\\alpha_n=-\\alpha_m + 0.5$")
ax_a.set_ylabel("Coeff var D")
ax_a.set_xlabel("$\\alpha_n$")
ax_a.legend()

ax_b.plot(*get_example_line_for_opposite_alpha_and_matching_phi(), color="#d63031", label="$\\phi=\\alpha_n$")
ax_b.plot(*get_example_line_for_opposite_alpha_and_higher_phi(), color="#000000", label="$\\phi=0.5*\\alpha_n + 0.5$")
ax_b.set_ylabel("Coeff var D")
ax_b.set_xlabel("$\\alpha_n, -\\alpha_m$")
ax_b.legend()

ax_b.set_xlim(0, 1)
ax_b.set_ylim(0, 1)

plt.tight_layout()
plt.show()


# Now draw examples
_, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, sharex="all", sharey="all")
random = numpy.random.Generator(numpy.random.MT19937(seed=1))
config = SimulationConfig(t_sim=750, random=random, track_n_vs_t=True)
T = (16.153070175438597, 3.2357834505600382)  # Based on measured values


def _plot_line(ax: Axes, results: SimulationResults):
    ax.plot(results.n_vs_t[:, 0], results.n_vs_t[:, 1] + results.n_vs_t[:, 2], color="black", alpha=0.5,
                 linewidth=2)
    if results.n_vs_t[-1, 1] + results.n_vs_t[-1, 2] == 0:  # Died
        ax.plot(results.n_vs_t[-1, 0], 0, "X", color="red")


# Left: low phi, low alpha
for _ in range(6):
    results = two_compartment_model.run_simulation(config, SimulationParameters.for_D_alpha_and_phi(
        D=30, phi=0.05, T=T, alpha_n=0, alpha_m=0))
    _plot_line(ax_left, results)

# Right: large phi, large alpha
for _ in range(6):
    results = two_compartment_model.run_simulation(config, SimulationParameters.for_D_alpha_and_phi(
        D=30, phi=0.95, T=T, alpha_n=0.95, alpha_m=-0.95))
    _plot_line(ax_right, results)

ax_left.set_ylabel("Proliferating cells")
ax_left.set_xlabel("Time (h)")

plt.show()
