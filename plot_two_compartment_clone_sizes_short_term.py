"""Plots an ordinary clone-size distribution on the short term of cells in the niche."""
import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from stem_cell_model import clone_size_simulator
from stem_cell_model.clone_size_simulator import CloneSizeSimulationConfig, TimedCloneSizeSimulationConfig
from stem_cell_model.parameters import SimulationParameters, SimulationConfig
from stem_cell_model.timed_clone_size_distributions import TimedCloneSizeDistribution
from stem_cell_model.two_compartment_model_space import run_simulation_niche


def _plot_final_clone_size(ax: Axes, results: TimedCloneSizeDistribution):
    clone_size_distribution = results.last()
    indices, heights = list(), list()
    for index, height in zip(clone_size_distribution.indices(), clone_size_distribution.to_height_array()):
        if index <= 1:  # Skip index 0 (if present) and 1
            continue
        indices.append(index)
        heights.append(height)

    ax.bar(indices, heights)
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_yticks([])
    ax.set_xlabel("Clone size")
    ax.set_ylabel("Number of clones")


def _add_title(ax: Axes, title: str):
    """Adds a title to the plot. Currently this method puts the title in the top right corner of the axes."""
    ax.text(0.96, 0.95, title, horizontalalignment='right',
                          verticalalignment='top', transform=ax.transAxes)

D = 30
T = (16.153070175438597, 3.2357834505600382)  # Based on measured values

parameters_symm_high_growth = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.95, alpha_m=-0.95, phi=0.95, T=T, a=1/T[0])
parameters_mixed_mid_growth = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.5, alpha_m=-0.5, phi=0.5, T=T, a=1/T[0])
parameters_asymm_low_growth = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.05, alpha_m=-0.05, phi=0.05, T=T, a=1/T[0])


random = numpy.random.Generator(numpy.random.MT19937(seed=1))
t_clone_size = 48
config = TimedCloneSizeSimulationConfig(t_clone_size=t_clone_size, t_interval=t_clone_size, random=random, n_crypts=1000)

fig, (ax_left, ax_middle, ax_right) = plt.subplots(1, 3, sharex="all")

# Left panel
results = clone_size_simulator.calculate_niche_over_time(
    run_simulation_niche, config, parameters_symm_high_growth)
_add_title(ax_left, "$\\alpha_n = 0.95$, $\\phi=0.95$")
_plot_final_clone_size(ax_left, results)

# Middle panel
results = clone_size_simulator.calculate_niche_over_time(
    run_simulation_niche, config, parameters_mixed_mid_growth)
_add_title(ax_middle, "$\\alpha_n = 0.5$, $\\phi=0.5$")
_plot_final_clone_size(ax_middle, results)


# Right panel
results = clone_size_simulator.calculate_niche_over_time(
    run_simulation_niche, config, parameters_asymm_low_growth)
_add_title(ax_right, "$\\alpha_n = 0.05$, $\\phi=0.05$")
_plot_final_clone_size(ax_right, results)


plt.tight_layout()
plt.show()
