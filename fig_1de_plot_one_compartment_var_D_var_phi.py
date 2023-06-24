import math
from typing import Dict, List

import numpy
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar

from stem_cell_model import sweeper, tools, two_compartment_model
import matplotlib.pyplot as plt

from stem_cell_model.parameters import SimulationConfig, SimulationParameters
from stem_cell_model.results import SimulationResults

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


def plot_overgrowth_for_S_against_phi(min_log_S: float, max_log_S: float):
    # Make S to y dict to support logarithmic plotting
    S_to_image_y = _get_S_to_image_y_dict(min_log_S, max_log_S)

    # Fill the image
    image = numpy.full((IMAGE_HEIGHT, STEPS_ALONG_PHI_AXIS), numpy.nan, dtype=numpy.float64)
    for params, multi_run_stats in sweeper.load_sweep_results("one_comp_sweep_data_var_D_var_phi"):
        image_ys = S_to_image_y[params.S]
        image_x = round(params.phi[0] * (STEPS_ALONG_PHI_AXIS - 1))
        statistics = tools.get_single_parameter_set_statistics(multi_run_stats)
        for image_y in image_ys:
            image[image_y, image_x] = statistics.n_explosions

    return image


fig, ((ax_top_left, ax_top_middle, ax_top_right), (ax_bottom_left, ax_bottom_middle, ax_bottom_right)) = plt.subplots(nrows=2, ncols=3, figsize=(6.498, 3.9),
                                                                                                                      gridspec_kw={"height_ratios": [0.94, 0.06]})
min_log_S = 0
max_log_S = 2

image_cov = plot_coeff_of_variation_for_S_against_phi(min_log_S, max_log_S)
image_cov_vmax = math.floor(numpy.nanmax(image_cov))

image_overgrowth = plot_overgrowth_for_S_against_phi(min_log_S, max_log_S)
image_overgrowth_vmax = 3  # Was: math.ceil(numpy.nanmax(image_overgrowth))

image_depletion = plot_depletion_for_S_against_phi(min_log_S, max_log_S)
image_depletion_vmax = 10  # Was: math.ceil(numpy.nanmax(image_depletion))

# Draw the depletion image
ax_top_left.set_title("Depletion rate / 1000h")
ax_top_left.set_facecolor("#b2bec3")
ax_top_left_image = ax_top_left.imshow(image_depletion, extent=(0, 1, max_log_S, min_log_S), aspect="auto", cmap=COLOR_MAP, interpolation="nearest", vmin=0, vmax=image_depletion_vmax)
ax_top_left.set_xlabel("$\\phi$")
ax_top_left.set_ylabel("$D$")
ax_top_left.set_yticks(numpy.linspace(min_log_S, max_log_S, 3))
ax_top_left.set_yticklabels([f"{10 ** log_S:.0f}" for log_S in numpy.linspace(min_log_S, max_log_S, 3)])

# Draw the overgrowth image
ax_top_middle.set_title("Overgrowth rate / 1000h")
ax_top_middle.set_facecolor("#000000")
ax_top_middle_image = ax_top_middle.imshow(image_overgrowth, extent=(0, 1, max_log_S, min_log_S), aspect="auto", cmap=COLOR_MAP, interpolation="nearest", vmin=0, vmax=image_overgrowth_vmax)
ax_top_middle.set_xlabel("$\\phi$")
ax_top_middle.set_ylabel("$D$")
ax_top_middle.set_yticks(numpy.linspace(min_log_S, max_log_S, 3))
ax_top_middle.set_yticklabels([f"{10 ** log_S:.0f}" for log_S in numpy.linspace(min_log_S, max_log_S, 3)])

# Draw the coefficient of variation image
ax_top_right.set_title("Coefficient of variation in $D(t)$")
ax_top_right.set_facecolor("#b2bec3")
ax_top_right_image = ax_top_right.imshow(image_cov, extent=(0, 1, max_log_S, min_log_S), aspect="auto", cmap=COLOR_MAP, interpolation="nearest", vmin=0, vmax=image_cov_vmax)
ax_top_right.set_xlabel("$\\phi$")
ax_top_right.set_ylabel("$D$")
ax_top_right.set_yticks(numpy.linspace(min_log_S, max_log_S, 3))
ax_top_right.set_yticklabels([f"{10 ** log_S:.0f}" for log_S in numpy.linspace(min_log_S, max_log_S, 3)])

fig.colorbar(ax_top_right_image, cax=ax_bottom_right, orientation="horizontal")
fig.colorbar(ax_top_middle_image, cax=ax_bottom_middle, orientation="horizontal")
fig.colorbar(ax_top_left_image, cax=ax_bottom_left, orientation="horizontal")
fig.tight_layout()
plt.show()


# Draw the example lines
ax_left: Axes
fig, (ax_left, ax_left_histogram, ax_right, ax_right_histogram) = plt.subplots(nrows=1, ncols=4, sharey="all",
                                                                             gridspec_kw={'width_ratios': [3, 1, 3, 1]})
random = numpy.random.Generator(numpy.random.MT19937(seed=1))
config = SimulationConfig(t_sim=260, random=random, track_n_vs_t=True)
T = (16.153070175438597, 3.2357834505600382)  # Based on measured values


def _plot_line(ax: Axes, results: SimulationResults):
    ax.plot(results.n_vs_t[:, 0], results.n_vs_t[:, 1] + results.n_vs_t[:, 2], color="black", alpha=0.5,
                 linewidth=2)
    if results.n_vs_t[-1, 1] + results.n_vs_t[-1, 2] == 0:  # Died
        ax.plot(results.n_vs_t[-1, 0], 0, "X", color="red")

def _plot_histogram(ax: Axes, config: SimulationConfig, params: SimulationParameters):
    bins = numpy.arange(0, 60, 1)
    counts = list()

    for i in range(2000):
        if i % 100 == 0:
            print(i)
        results = two_compartment_model.run_simulation(config, params)
        last_count = results.n_vs_t[-1, 1] + results.n_vs_t[-1, 2]
        counts.append(last_count)
    ax.hist(counts, orientation="horizontal", bins=bins, color="black")
    ax.axis("off")

# Left: small phi
params = SimulationParameters.for_one_compartment(D=15, phi=0.95, T=T)
for _ in range(6):
    results = two_compartment_model.run_simulation(config, params)
    _plot_line(ax_left, results)
_plot_histogram(ax_left_histogram, config, params)

# Right: large phi
params = SimulationParameters.for_one_compartment(D=15, phi=0.05, T=T)
for _ in range(6):
    results = two_compartment_model.run_simulation(config, params)
    _plot_line(ax_right, results)
_plot_histogram(ax_right_histogram, config, params)

ax_left.set_ylabel("Proliferating cells")
ax_left.set_xlabel("Time (h)")

plt.show()
