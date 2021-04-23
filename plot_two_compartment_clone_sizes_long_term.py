"""Different clone size distributions, short term. Comparing variability."""
import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from stem_cell_model import clone_size_simulator
from stem_cell_model.clone_size_simulator import CloneSizeSimulationConfig, TimedCloneSizeSimulationConfig
from stem_cell_model.parameters import SimulationParameters, SimulationConfig
from stem_cell_model.timed_clone_size_distributions import TimedCloneSizeDistribution
from stem_cell_model.two_compartment_model_space import run_simulation_niche


def _plot_clone_sizes_over_time(ax: Axes, results: TimedCloneSizeDistribution, *, legend: bool = True):
    durations = results.get_durations()
    ax.plot(durations, results.get_clone_size_counts([2]), label="2", color="#00b894")
    ax.plot(durations, results.get_clone_size_counts([3, 4]), label="3-4", color="#fdcb6e")
    ax.plot(durations, results.get_clone_size_counts([5, 6, 7, 8]), label="5-8", color="#a29bfe")
    ax.plot(durations, results.get_clone_size_counts(range(9, 17)), label="9-16", color="#fd79a8")
    ax.plot(durations, results.get_clone_size_counts(range(17, 33)), label="17-32", color="#d63031")
    ax.plot(durations, results.get_clone_size_counts(range(33, 65)), label="33-64", color="#0984e3")
    ax.plot(durations, results.get_clone_size_counts(range(65, 129)), label="65-128", color="#ff7675")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Number of clones")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Number of clones")

    if legend:
        ax.legend()


D = 30
T = (16.153070175438597, 3.2357834505600382)  # Based on measured values

parameters_symm_low_noise = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.95, alpha_m=-0.95, phi=0.95, T=T, a=1/T[0])
parameters_symm_high_noise = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.05, alpha_m=-0.05, phi=0.95, T=T, a=1/T[0])
parameters_asymm = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.05, alpha_m=-0.05, phi=0.05, T=T, a=1/T[0])
parameters_mixed = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.05, alpha_m=-0.05, phi=0.5, T=T, a=1/T[0])

random = numpy.random.Generator(numpy.random.MT19937(seed=1))
t_clone_size = 300
t_interval = 1
config = TimedCloneSizeSimulationConfig(t_clone_size=t_clone_size, t_interval=t_interval, random=random, n_crypts=500)

fig, ((ax_top_left, ax_top_right), (ax_bottom_left, ax_bottom_right)) = plt.subplots(2, 2)

# Top left panel
results = clone_size_simulator.calculate_proliferative_over_time(
    run_simulation_niche, config, parameters_symm_low_noise)
ax_top_left.set_title("Symmetric low noise ($\\alpha_n = 0.95$, $\\phi=0.95$)")
_plot_clone_sizes_over_time(ax_top_left, results)

# Bottom left panel
results = clone_size_simulator.calculate_proliferative_over_time(
    run_simulation_niche, config, parameters_symm_high_noise)
ax_bottom_left.set_title("Symmetric high noise ($\\alpha_n = 0.05$, $\\phi=0.95$)")
_plot_clone_sizes_over_time(ax_bottom_left, results, legend=False)

# Top right panel
results = clone_size_simulator.calculate_proliferative_over_time(
    run_simulation_niche, config, parameters_asymm)
ax_top_right.set_title("Asymmetric ($\\alpha_n = 0.05$, $\\phi=0.05$)")
_plot_clone_sizes_over_time(ax_top_right, results, legend=False)

# Right bottom panel
results = clone_size_simulator.calculate_proliferative_over_time(
    run_simulation_niche, config, parameters_mixed)
ax_bottom_right.set_title("Mixed symmetric and asymmetric ($\\alpha_n = 0.05$, $\\phi=0.5$)")
_plot_clone_sizes_over_time(ax_bottom_right, results, legend=False)

plt.tight_layout()
plt.show()
