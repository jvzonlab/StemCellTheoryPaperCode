import math
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, NamedTuple

import matplotlib
import numpy
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from numpy import ndarray

from stem_cell_model import sweeper, tools, two_compartment_model
import matplotlib.pyplot as plt

from stem_cell_model.parameters import SimulationConfig, SimulationParameters
from stem_cell_model.results import SimulationResults

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

STEPS_ALONG_AXIS = 40
COLOR_MAP = "gnuplot"


class _AlphaAndPhi(NamedTuple):
    alpha_n: float
    alpha_m: float
    phi_n: float
    phi_m: float

    def matches(self, params: SimulationParameters) -> bool:
        return abs(params.alpha[0] - self.alpha_n) < 0.001 and \
            abs(params.alpha[1] - self.alpha_m) < 0.001 and \
            abs(params.phi[0] - self.phi_n) < 0.001 and \
            abs(params.phi[1] - self.phi_m) < 0.001


def main():
    # Plot <D> als functie van S gegeven alpha_n en alpha_m
    low_alpha_high_phi = _AlphaAndPhi(alpha_n=0.2, alpha_m=-0.2, phi_n=0.95, phi_m=0.95)
    high_alpha_n_and_m = _AlphaAndPhi(alpha_n=0.95, alpha_m=-0.95, phi_n=0.95, phi_m=0.95)

    keys = [low_alpha_high_phi, high_alpha_n_and_m]
    D_values = defaultdict(list)
    D_values_stdev = defaultdict(list)
    S_values = defaultdict(list)

    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_variable_S"):
        statistics = tools.get_single_parameter_set_statistics(multi_run_stats)

        for key in keys:
            if key.matches(params):
                D_values[key].append(statistics.d_mean)
                D_values_stdev[key].append(statistics.d_std)
                S_values[key].append(params.S)

    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_fixed_D"):
        statistics = tools.get_single_parameter_set_statistics(multi_run_stats)

        for key in keys:
            if key.matches(params):
                D_values[key].append(statistics.d_mean)
                D_values_stdev[key].append(statistics.d_std)
                S_values[key].append(params.S)


    _sort(D_values, D_values_stdev, S_values)

    figure = plt.figure()
    ax = figure.gca()
    ax.set_xlabel("S")
    ax.set_ylabel("D")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 420)
    _plot_D_by_S(ax, low_alpha_high_phi, D_values, D_values_stdev, S_values, color="#f2bb00")
    _plot_D_by_S(ax, high_alpha_n_and_m, D_values, D_values_stdev, S_values, color="#2ca02c")
    ax.legend()
    plt.show()


def _plot_D_by_S(ax: Axes, alpha_phi: _AlphaAndPhi, D_values: Dict[_AlphaAndPhi, ndarray],
                 D_values_stdev: Dict[_AlphaAndPhi, ndarray], S_values: Dict[_AlphaAndPhi, ndarray], color: str):
    D_predicted = numpy.log(1 + alpha_phi.alpha_n) * S_values[alpha_phi] * (alpha_phi.alpha_m - alpha_phi.alpha_n) / alpha_phi.alpha_m
    ax.plot(S_values[alpha_phi], D_predicted, color=color, linewidth=1)
    ax.errorbar(S_values[alpha_phi], D_values[alpha_phi], yerr=D_values_stdev[alpha_phi], marker="o", markersize=5,
                color=color, linewidth=0, elinewidth=1,
                label=f"$\\alpha_n=${alpha_phi.alpha_n}, $\\alpha_m=${alpha_phi.alpha_m}, $\\phi=${alpha_phi.phi_n}")


def _sort(D_values: Dict[_AlphaAndPhi, List[float]], D_values_stdev: Dict[_AlphaAndPhi, List[float]], S_values: Dict[_AlphaAndPhi, List[float]]):
    for key in D_values.keys():
        D_values_for_key = numpy.array(D_values[key])
        sorting = numpy.argsort(D_values_for_key)
        D_values[key] = D_values_for_key[sorting]
        D_values_stdev[key] = numpy.array(D_values_stdev[key])[sorting]
        S_values[key] = numpy.array(S_values[key])[sorting]


if __name__ == "__main__":
    main()
