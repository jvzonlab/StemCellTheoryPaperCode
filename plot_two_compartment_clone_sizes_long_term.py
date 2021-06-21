"""Different clone size distributions, short term. Comparing variability."""
import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from stem_cell_model import clone_size_simulator
from stem_cell_model.clone_size_simulator import CloneSizeSimulationConfig, TimedCloneSizeSimulationConfig
from stem_cell_model.parameters import SimulationParameters, SimulationConfig
from stem_cell_model.timed_clone_size_distributions import TimedCloneSizeDistribution
from stem_cell_model.two_compartment_model_space import run_simulation_niche


def _plot_clone_sizes_over_time(ax: Axes, results: TimedCloneSizeDistribution, params: SimulationParameters):
    durations = results.get_durations()
    ax.plot(durations, results.get_average_clone_size_over_time(), color="#fdcb6e")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Average clone size")
    if params.S <= 30:
        ax.axhline(y=params.S, color="#dddddd")


def _add_title(ax: Axes, title: str):
    """Adds a title to the plot. Currently this method puts the title in the top right corner of the axes."""
    ax.text(0.96, 0.95, title, horizontalalignment='right',
                          verticalalignment='top', transform=ax.transAxes)

D = 30
T = (16.153070175438597, 3.2357834505600382)  # Based on measured values

parameters_symm_high_growth = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.95, alpha_m=-0.95, phi=0.95, T=T, a=10/T[0])
parameters_symm_mid_growth = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.5, alpha_m=-0.5, phi=0.95, T=T, a=10/T[0])
parameters_symm_low_growth = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.05, alpha_m=-0.05, phi=0.95, T=T, a=10/T[0])
parameters_mixed_mid_growth = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.5, alpha_m=-0.5, phi=0.5, T=T, a=10/T[0])
parameters_mixed_low_growth = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.05, alpha_m=-0.05, phi=0.5, T=T, a=10/T[0])
parameters_asymm_low_growth = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.05, alpha_m=-0.05, phi=0.05, T=T, a=10/T[0])


random = numpy.random.Generator(numpy.random.MT19937(seed=1))
t_clone_size = 24 * 7 * 2
t_interval = 24
config = TimedCloneSizeSimulationConfig(t_clone_size=t_clone_size, t_interval=t_interval, random=random, n_crypts=1000)

fig, ((ax_bottom_left, ax_bottom_middle, ax_bottom_right), (ax_middle_left, ax_middle_middle, ax_middle_right), (ax_top_left, ax_top_middle, ax_top_right)) = plt.subplots(3, 3, sharex="all", sharey="all")

# Top left panel
results = clone_size_simulator.calculate_proliferative_in_niche_over_time(
    run_simulation_niche, config, parameters_symm_low_growth)
_add_title(ax_top_left, "$\\alpha_n = 0.05$, $\\phi=0.95$")
_plot_clone_sizes_over_time(ax_top_left, results, parameters_symm_low_growth)

# Top middle panel
results = clone_size_simulator.calculate_proliferative_in_niche_over_time(
    run_simulation_niche, config, parameters_symm_mid_growth)
ax_top_middle.set_xlabel("n/<n(t)>")
_add_title(ax_top_middle, "$\\alpha_n = 0.5$, $\\phi=0.95$")
_plot_clone_sizes_over_time(ax_top_middle, results, parameters_symm_mid_growth)

# Top right panel
results = clone_size_simulator.calculate_proliferative_in_niche_over_time(
    run_simulation_niche, config, parameters_symm_high_growth)
_add_title(ax_top_right, "$\\alpha_n = 0.95$, $\\phi=0.95$")
_plot_clone_sizes_over_time(ax_top_right, results, parameters_symm_high_growth)

# Middle left panel
results = clone_size_simulator.calculate_proliferative_in_niche_over_time(
    run_simulation_niche, config, parameters_mixed_low_growth)
_add_title(ax_middle_left, "$\\alpha_n = 0.05$, $\\phi=0.5$")
ax_middle_left.set_ylabel("<n(t)> P_n(t)")
_plot_clone_sizes_over_time(ax_middle_left, results, parameters_mixed_low_growth)

# Middle middle panel
results = clone_size_simulator.calculate_proliferative_in_niche_over_time(
    run_simulation_niche, config, parameters_mixed_mid_growth)
_add_title(ax_middle_middle, "$\\alpha_n = 0.5$, $\\phi=0.5$")
_plot_clone_sizes_over_time(ax_middle_middle, results, parameters_mixed_mid_growth)

# Middle right panel
ax_middle_right.set_axis_off()

# Bottom left panel
results = clone_size_simulator.calculate_proliferative_in_niche_over_time(
    run_simulation_niche, config, parameters_asymm_low_growth)
_add_title(ax_bottom_left, "$\\alpha_n = 0.05$, $\\phi=0.05$")
_plot_clone_sizes_over_time(ax_bottom_left, results, parameters_asymm_low_growth)

# Bottom middle panel (not visible)
ax_bottom_middle.set_axis_off()

# Bottom right panel (not visible)
ax_bottom_right.set_axis_off()

plt.tight_layout()
plt.show()
