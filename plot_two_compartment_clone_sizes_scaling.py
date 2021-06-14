
import numpy
import matplotlib.cm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.axis import Axis

from stem_cell_model import clone_size_simulator
from stem_cell_model.clone_size_simulator import TimedCloneSizeSimulationConfig
from stem_cell_model.parameters import SimulationParameters, SimulationConfig
from stem_cell_model.timed_clone_size_distributions import TimedCloneSizeDistribution
from stem_cell_model.two_compartment_model import run_simulation

_COLORS = [matplotlib.cm.Blues(x) for x in [0.1, 0.4, 0.7, 1.0]]


def _plot_clone_scaling_over_time(ax: Axes, results: TimedCloneSizeDistribution, *, legend: bool = True, predicted_scaling: bool = False):
    clone_sizes = results.get_clone_sizes()

    durations = results.get_durations()
    average_clone_size = results.get_average_clone_size_over_time()
    clone_count = results.get_clone_count_over_time()
    # Plot x: n/<n(t)>
    # Plot y: <n(t)> P_n(t) = average clone size * [clone size fraction of size n]

    for i, time in enumerate(durations):
        x = clone_sizes / average_clone_size[i]
        size_fraction = results.get_distribution_at(i).get_clone_size_frequencies(clone_sizes) / clone_count[i]
        y = average_clone_size[i] * size_fraction
        ax.scatter(x, y, label=f"{time/24:.0f} days", s=5, color=_COLORS[i])

    # Plot predicted scaling function [Lopez-Garcia2010]
    if predicted_scaling:
        x = numpy.arange(0, 5, 0.01)
        F = (numpy.pi * x / 2) * numpy.exp(-numpy.pi * x ** 2 / 4)
        ax.plot(x, F, label=f"Scaling", color="black")

    ax.set_xlabel("n/<n(t)>")
    ax.set_ylabel("<n(t)> P_n(t)")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)

    if legend:
        ax.legend()


D = 30
T = (16.153070175438597, 3.2357834505600382)  # Based on measured values

parameters_symm_high_growth = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.95, alpha_m=-0.95, phi=0.95, T=T, a=1/T[0])
parameters_symm_mid_growth = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.5, alpha_m=-0.5, phi=0.95, T=T, a=1/T[0])
parameters_symm_low_growth = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.05, alpha_m=-0.05, phi=0.95, T=T, a=1/T[0])
parameters_mixed_mid_growth = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.5, alpha_m=-0.5, phi=0.5, T=T, a=1/T[0])
parameters_mixed_low_growth = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.05, alpha_m=-0.05, phi=0.5, T=T, a=1/T[0])
parameters_asymm_low_growth = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.05, alpha_m=-0.05, phi=0.05, T=T, a=1/T[0])


random = numpy.random.Generator(numpy.random.MT19937(seed=1))
t_clone_size = 24 * 7 * 4
t_interval = 24 * 7
config = TimedCloneSizeSimulationConfig(t_clone_size=t_clone_size, t_interval=t_interval, random=random, n_crypts=1000)

fig, ((ax_top_left, ax_top_middle, ax_top_right), (ax_middle_left, ax_middle_middle, ax_middle_right), (ax_bottom_left, ax_bottom_middle, ax_bottom_right)) = plt.subplots(3, 3)

# Top left panel
results = clone_size_simulator.calculate_proliferative_in_niche_over_time(
    run_simulation, config, parameters_symm_high_growth)
ax_top_left.set_title("Symmetric high growth ($\\alpha_n = 0.95$, $\\phi=0.95$)")
_plot_clone_scaling_over_time(ax_top_left, results, predicted_scaling=True)

# Top middle panel
results = clone_size_simulator.calculate_proliferative_in_niche_over_time(
    run_simulation, config, parameters_symm_mid_growth)
ax_top_middle.set_title("Symmetric mid growth ($\\alpha_n = 0.5$, $\\phi=0.95$)")
_plot_clone_scaling_over_time(ax_top_middle, results, legend=False)

# Top right panel
results = clone_size_simulator.calculate_proliferative_in_niche_over_time(
    run_simulation, config, parameters_symm_low_growth)
ax_top_right.set_title("Symmetric low growth ($\\alpha_n = 0.05$, $\\phi=0.95$)")
_plot_clone_scaling_over_time(ax_top_right, results, legend=False)

ax_middle_left.set_axis_off()

# Middle middle panel
results = clone_size_simulator.calculate_proliferative_in_niche_over_time(
    run_simulation, config, parameters_mixed_mid_growth)
ax_middle_middle.set_title("Mixed mid growth ($\\alpha_n = 0.5$, $\\phi=0.5$)")
_plot_clone_scaling_over_time(ax_middle_middle, results, legend=False)

# Middle right panel
results = clone_size_simulator.calculate_proliferative_in_niche_over_time(
    run_simulation, config, parameters_mixed_low_growth)
ax_middle_right.set_title("Mixed low growth ($\\alpha_n = 0.05$, $\\phi=0.5$)")
_plot_clone_scaling_over_time(ax_middle_right, results, legend=False)

ax_bottom_left.set_axis_off()
ax_bottom_middle.set_axis_off()

# Bottom right panel
results = clone_size_simulator.calculate_niche_over_time(
    run_simulation, config, parameters_asymm_low_growth)
ax_bottom_right.set_title("Asymmetric low growth ($\\alpha_n = 0.05$, $\\phi=0.05$)")
_plot_clone_scaling_over_time(ax_bottom_right, results, legend=False)

plt.tight_layout()
plt.show()
