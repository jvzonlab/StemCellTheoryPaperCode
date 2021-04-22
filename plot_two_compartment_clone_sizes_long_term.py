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
    ax.plot(durations, results.get_clone_size_counts(2, 2), label="2", color="#00b894")
    ax.plot(durations, results.get_clone_size_counts(3, 4), label="3-4", color="#fdcb6e")
    ax.plot(durations, results.get_clone_size_counts(5, 8), label="5-8", color="#a29bfe")
    ax.plot(durations, results.get_clone_size_counts(9, 16), label="9-16", color="#fd79a8")
    ax.plot(durations, results.get_clone_size_counts(17, 32), label="17-32", color="#d63031")
    ax.plot(durations, results.get_clone_size_counts(33, 64), label="33-64", color="#0984e3")
    ax.plot(durations, results.get_clone_size_counts(65, 128), label="65-128", color="#ff7675")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Number of clones")

    if legend:
        ax.legend()


D = 30
T = (16.153070175438597, 3.2357834505600382)  # Based on measured values

parameters_low_noise = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.95, alpha_m=-0.95, phi=0.95, T=T, a=1/T[0])
parameters_high_noise = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.05, alpha_m=-0.05, phi=0.95, T=T, a=1/T[0])

random = numpy.random.Generator(numpy.random.MT19937(seed=1))
t_clone_size = 300
t_interval = 1
config = TimedCloneSizeSimulationConfig(t_clone_size=t_clone_size, t_interval=t_interval, random=random, n_crypts=500)

fig, (ax_top, ax_bottom) = plt.subplots(2, 1)

# Top panel
results_low_noise = clone_size_simulator.calculate_proliferative_over_time(
    run_simulation_niche, config, parameters_low_noise)
ax_top.set_title("Low noise ($\\alpha_n = 0.95$)")
_plot_clone_sizes_over_time(ax_top, results_low_noise)

# Bottom panel
results_high_noise = clone_size_simulator.calculate_proliferative_over_time(
    run_simulation_niche, config, parameters_high_noise)
ax_bottom.set_title("High noise ($\\alpha_n = 0.05$)")
_plot_clone_sizes_over_time(ax_bottom, results_high_noise, legend=False)

plt.tight_layout()
plt.show()
