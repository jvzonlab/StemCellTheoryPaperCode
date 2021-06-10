"""Used to plot how rearrangement rates change the stability of the system"""
import math
from typing import Dict, List

import matplotlib
import numpy
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar

from stem_cell_model import sweeper, tools
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

STEPS_ALONG_AXIS = 40
COLOR_MAP = "gnuplot"

def plot_cov_against_rearrangement(ax: Axes):
    x_values = list()
    y_values = list()
    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_variable_mixing"):
        aT = params.a * params.T[0]

        statistics = tools.get_single_parameter_set_statistics(multi_run_stats)
        x_values.append(aT)
        y_values.append(statistics.d_coeff_var)

    ax.scatter(x_values, y_values)

figure = plt.figure()
ax: Axes = figure.gca()
ax.set_xscale("log")
ax.set_xlabel("$r \\cdot T$")
ax.set_ylabel("Coeff of var in $D(t)$")
plot_cov_against_rearrangement(ax)
plt.show()
