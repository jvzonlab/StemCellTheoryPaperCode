"""Plots an ordinary clone-size distribution on the short term of cells in the niche."""
import itertools

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from stem_cell_model import clone_size_simulator
from stem_cell_model.clone_size_distributions import CloneSizeDistribution
from stem_cell_model.clone_size_simulator import CloneSizeSimulationConfig
from stem_cell_model.parameters import SimulationParameters
from stem_cell_model.two_compartment_model import run_simulation
from stem_cell_model.two_compartment_model_neutral_drift import run_simulation_neutral_drift


def _plot_final_clone_size(ax: Axes, clone_size_distribution: CloneSizeDistribution):
    indices, heights = list(), list()
    for index, height in zip(clone_size_distribution.indices(), clone_size_distribution.to_height_array()):
        if index <= 1:  # Skip index 0 (if present) and 1
            continue
        indices.append(index)
        heights.append(height)

    ax.bar(indices, heights)
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_xlabel("Clone size")
    ax.set_ylabel("Number of clones")


def _add_title(ax: Axes, title: str):
    """Adds a title to the plot. Currently this method puts the title in the top right corner of the axes."""
    ax.text(0.96, 0.95, title, horizontalalignment='right',
                          verticalalignment='top', transform=ax.transAxes)


T = (16.153070175438597, 3.2357834505600382)  # Based on measured values


random = numpy.random.Generator(numpy.random.MT19937(seed=1))
t_clone_size = 48
config = CloneSizeSimulationConfig(t_clone_size=t_clone_size, random=random, n_crypts=1000)

fig, axes = plt.subplots(2, 3, sharex="all")
axes = list(itertools.chain(*axes))  # This flattens the axes list

for ax, S in zip(axes, [1, 2, 5, 10, 20, 30]):
    results = clone_size_simulator.calculate(
        run_simulation_neutral_drift, config, SimulationParameters.for_neutral_drift(S=S, T=T))
    _add_title(ax, f"S={S}")
    _plot_final_clone_size(ax, results)


plt.tight_layout()
plt.show()
