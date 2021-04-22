"""Different clone size distributions, short term. Comparing variability."""
import numpy
from matplotlib import pyplot as plt

from stem_cell_model import clone_size_simulator
from stem_cell_model.clone_size_simulator import CloneSizeSimulationConfig, TimedCloneSizeSimulationConfig
from stem_cell_model.parameters import SimulationParameters, SimulationConfig
from stem_cell_model.two_compartment_model_space import run_simulation_niche

D = 30
T = (16.153070175438597, 3.2357834505600382)  # Based on measured values

parameters = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.95, alpha_m=-0.95, phi=0.95, T=T, a=1/T[0])

random = numpy.random.Generator(numpy.random.MT19937(seed=1))
t_clone_size = 200
t_interval = 2
config = TimedCloneSizeSimulationConfig(t_clone_size=t_clone_size, t_interval=t_interval, random=random, n_crypts=1)

fig, (ax_left, ax_right) = plt.subplots(1, 2)

results = clone_size_simulator.calculate_proliferative_over_time(
    run_simulation_niche, config, parameters)
ax_left.set_title("Low noise ($\\alpha_n = 0.95$)")
ax_left.plot(results.get_times(), results.get_clone_size_counts(2, 2), label="2")
ax_left.plot(results.get_times(), results.get_clone_size_counts(3, 4), label="3-4")
ax_left.plot(results.get_times(), results.get_clone_size_counts(5, 8), label="5-8")
ax_left.plot(results.get_times(), results.get_clone_size_counts(9, 16), label="9-16")
ax_left.plot(results.get_times(), results.get_clone_size_counts(17, 32), label="17-32")
ax_left.plot(results.get_times(), results.get_clone_size_counts(33, 64), label="33-64")
ax_left.plot(results.get_times(), results.get_clone_size_counts(65, 128), label="65-128")
ax_left.set_xlabel("Time (h)")
ax_left.legend()

niche_config = config.to_niche_config()
niche_config.track_n_vs_t = True
example_results = run_simulation_niche(niche_config, parameters)

example_results.lineages.draw_lineages(ax_right, config.t_clone_size, 0)
ax_right.invert_yaxis()


plt.show()
