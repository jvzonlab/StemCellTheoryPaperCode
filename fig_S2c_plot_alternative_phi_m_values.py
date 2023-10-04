"""Used to give a general overview of the two compartment model"""
import math
from collections import OrderedDict
from typing import Dict, List, Tuple

import matplotlib
import numpy
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar

from stem_cell_model import sweeper, tools, two_compartment_model
import matplotlib.pyplot as plt

from stem_cell_model.parameters import SimulationConfig, SimulationParameters
from stem_cell_model.results import SimulationResults, MultiRunStats

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

STEPS_ALONG_AXIS = 40
COLOR_MAP = "gnuplot"

T = (16.153070175438597, 3.2357834505600382)  # Based on measured values
D = 30  # average number of dividing cells
CONFIG = SimulationConfig(t_sim=int(1e5), random=numpy.random.Generator(numpy.random.MT19937(seed=1)))


def get_example_line_for_matching_phi() -> Tuple[List[float], List[float]]:
    """Line along the lowest phi for every alpha. This is one of the optimal lines for minimizing fluctuations.."""
    values = dict()
    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_fixed_D"):
        if params.alpha[0] == -params.alpha[1] and params.phi[0] == params.alpha[0]:
            statistics = tools.get_single_parameter_set_statistics(multi_run_stats)
            values[params.alpha[0]] = statistics.d_coeff_var

    x_values = list()
    y_values = list()
    for key, value in sorted(values.items()):
        x_values.append(key)
        y_values.append(value)
    return x_values, y_values


def get_example_line_for_higher_phi_m() -> Tuple[List[float], List[float]]:
    """Same as above, but with phi_m = mean(1, phi_n)"""

    alpha_values = list()
    coefficients_of_variation = list()
    for alpha in numpy.linspace(1 / STEPS_ALONG_AXIS, 1, STEPS_ALONG_AXIS):
        params = SimulationParameters.for_D_alpha_and_phi_ext(D=D, alpha_n=alpha, alpha_m=-alpha, phi_n=alpha, phi_m=(1 + alpha) / 2, T=T)
        if params is None:
            continue

        # Run simulation
        output = MultiRunStats()
        while output.t_tot < CONFIG.t_sim:
            config = SimulationConfig(t_sim=CONFIG.t_sim - output.t_tot, random=CONFIG.random)
            output.add_results(two_compartment_model.run_simulation(config, params))
        stats = tools.get_single_parameter_set_statistics(output)

        alpha_values.append(alpha)
        coefficients_of_variation.append(stats.d_coeff_var)
    return alpha_values, coefficients_of_variation


def get_example_line_for_one_phi_m() -> Tuple[List[float], List[float]]:
    """Same as above, but with phi_m = 1"""

    alpha_values = list()
    coefficients_of_variation = list()
    for alpha in numpy.linspace(1 / STEPS_ALONG_AXIS, 1, STEPS_ALONG_AXIS):
        params = SimulationParameters.for_D_alpha_and_phi_ext(D=D, alpha_n=alpha, alpha_m=-alpha, phi_n=alpha, phi_m=1, T=T)
        if params is None:
            continue

        # Run simulation
        output = MultiRunStats()
        while output.t_tot < CONFIG.t_sim:
            config = SimulationConfig(t_sim=CONFIG.t_sim - output.t_tot, random=CONFIG.random)
            output.add_results(two_compartment_model.run_simulation(config, params))
        stats = tools.get_single_parameter_set_statistics(output)

        alpha_values.append(alpha)
        coefficients_of_variation.append(stats.d_coeff_var)
    return alpha_values, coefficients_of_variation


# Now draw values along lines
fig = plt.figure(figsize=(4.181, 3))
ax_b = fig.gca()

ax_b.plot(*get_example_line_for_one_phi_m(), color="#000000", label="$\\phi_n=\\alpha_n, \\phi_m=1$")
ax_b.plot(*get_example_line_for_higher_phi_m(), color="#1cbdbd", label="$\\phi_n=\\alpha_n, \\phi_m=(1 + \\alpha_n) / 2$")
ax_b.plot(*get_example_line_for_matching_phi(), color="#d63031", label="$\\phi_n=\\alpha_n, \\phi_m=\\alpha_n$")
ax_b.set_ylabel("Coeff var D")
ax_b.set_xlabel("$\\alpha_n, -\\alpha_m$")
ax_b.legend()

ax_b.set_xlim(0, 1)
ax_b.set_ylim(0, 1)

plt.tight_layout()
plt.show()

