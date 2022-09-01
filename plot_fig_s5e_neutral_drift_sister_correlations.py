import itertools

import matplotlib
import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from stem_cell_model.parameters import SimulationParameters, SimulationConfig
from stem_cell_model.two_compartment_model_neutral_drift import run_simulation_neutral_drift
import matplotlib.colors

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'  # export text as text in SVG, not as paths


def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(numpy.linspace(minval, maxval, n)))
    return new_cmap



def _add_title(ax: Axes, title: str):
    """Adds a title to the plot. Currently this method puts the title in the top right corner of the axes."""
    ax.text(1, 1, title, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)


T = (16.153070175438597, 3.2357834505600382)  # Based on measured values
COLORMAP = _truncate_colormap(plt.get_cmap("magma_r"), 0, 0.75)


random = numpy.random.Generator(numpy.random.MT19937(seed=1))
t_measurement = 100000
config = SimulationConfig(
        t_sim=t_measurement,
        random=random,
        track_lineage_time_interval=(0, t_measurement))

fig, axes = plt.subplots(2, 3, sharex="all", sharey="all")
axes = list(itertools.chain(*axes))  # This flattens the axes list

for ax, S in zip(axes, [1, 2, 5, 10, 30, 60]):
    ax: Axes
    results = run_simulation_neutral_drift(config, SimulationParameters.for_neutral_drift(S=S, T=T))
    _add_title(ax, f"S={S}")
    division_counts = results.lineages.count_divisions()

    array = numpy.array([[division_counts.sisters_symmetric_dividing, numpy.nan],
             [division_counts.sisters_asymmetric, division_counts.sisters_symmetric_non_dividing]], dtype=numpy.float64)
    array /= numpy.nansum(array)

    # Draw heatmap
    ax.imshow(array, cmap=COLORMAP, vmin = 0.03, vmax=0.41)

    # Draw text
    for x in [0, 1]:
        for y in [0, 1]:
            value = array[y, x]
            if numpy.isnan(value):
                continue
            color = "white" if value > 0.2 else "black"
            ax.text(x, y, f"{array[y, x]:.2f}", horizontalalignment='center', verticalalignment='center', color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["divides", "never\ndivides"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["divides", "never\ndivides"])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


plt.tight_layout()
plt.show()
