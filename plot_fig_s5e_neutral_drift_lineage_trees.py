import itertools

import matplotlib
import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from stem_cell_model.parameters import SimulationParameters, SimulationConfig
from stem_cell_model.two_compartment_model_neutral_drift import run_simulation_neutral_drift

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'  # export text as text in SVG, not as paths


def _add_title(ax: Axes, title: str):
    """Adds a title to the plot. Currently this method puts the title in the top right corner of the axes."""
    ax.text(0.96, 0.95, title, horizontalalignment='right',
                          verticalalignment='top', transform=ax.transAxes)


T = (16.153070175438597, 3.2357834505600382)  # Based on measured values


random = numpy.random.Generator(numpy.random.MT19937(seed=1))
t_clone_size = 50
config = SimulationConfig(
        t_sim=t_clone_size,
        random=random,
        track_lineage_time_interval=(0, t_clone_size))

fig, axes = plt.subplots(2, 3, sharex="all", sharey="all")
axes = list(itertools.chain(*axes))  # This flattens the axes list

for ax, S in zip(axes, [1, 2, 5, 10, 30, 60]):
    results = run_simulation_neutral_drift(config, SimulationParameters.for_neutral_drift(S=S, T=T))
    _add_title(ax, f"S={S}")
    results.lineages.draw_lineages(ax, int(config.t_sim))
    ax.set_ylim(config.t_sim + 1, -1)
    ax.set_xlim(-1, 30)


plt.tight_layout()
plt.show()
