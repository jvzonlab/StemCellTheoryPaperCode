"""Script to quickly plot the lineages of a certain parameter set."""


import matplotlib.pyplot as plt
import numpy
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from stem_cell_model import clone_size_distributions, timed_clone_size_distributions
from stem_cell_model.parameters import SimulationParameters, SimulationConfig
from stem_cell_model.two_compartment_model import run_simulation

D = 3
T = (16.153070175438597, 3.2357834505600382)  # Based on measured values
random = numpy.random.Generator(numpy.random.MT19937(seed=1))
params = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.05, alpha_m=-0.05, phi=0.95, T=T, a=1 / T[0])
print("D", params.D, ",  S", params.S)
t_sim = 100
config = SimulationConfig(t_sim=t_sim, n_max=1000, random=random, track_lineage_time_interval=(0, t_sim), track_n_vs_t=True)

results = run_simulation(config, params)

figure, (ax_top, ax_bottom) = plt.subplots(1, 2)

results.lineages.draw_lineages(ax_top, config.t_sim)
ax_top.invert_yaxis()

clone_sizes = timed_clone_size_distributions.get_niche_clone_size_distribution(results.lineages, 0, config.t_sim, config.t_sim).last()
print("Total cells recorded in niche: ", clone_sizes.get_average() * clone_sizes.get_clone_count())
ax_bottom.bar(clone_sizes.indices(), clone_sizes.to_height_array())

# time = results.n_vs_t[:, 0]
# S_over_time = results.n_vs_t[:, 1] + results.u_vs_t[:, 1]
# print(S_over_time)
# ax_bottom.plot(time, S_over_time)

plt.show()
