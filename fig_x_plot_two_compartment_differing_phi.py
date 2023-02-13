from typing import List

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray

from stem_cell_model import tools, sweeper
from stem_cell_model.parameters import SimulationParameters
from stem_cell_model.results import MultiRunStats
from stem_cell_model.tools import SingleParameterSetStatistics


class _SimulationForPoint:
    alpha_n: float
    alpha_m: float

    phi_n: ndarray  # Linearly increasing phi_n
    phi_m: ndarray  # Linearly increasing phi_m
    cov_of_variation: ndarray  # Indexed as phi_n, phi_m
    f_collapse: ndarray

    def __init__(self, *, alpha_n: float, alpha_m: float, steps_along_axis: int = 40):
        self.phi_n = numpy.linspace(25, 1000, steps_along_axis) * 0.001
        self.phi_m = numpy.linspace(25, 1000, steps_along_axis) * 0.001
        self.cov_of_variation = numpy.full((self.phi_n.shape[0], self.phi_m.shape[0]), numpy.nan)
        self.f_collapse = numpy.copy(self.cov_of_variation)

        self.alpha_n = alpha_n
        self.alpha_m = alpha_m

    def offer_data_point(self, params: SimulationParameters, multi_run_stats: MultiRunStats):
        if abs(params.alpha[0] - self.alpha_n) > 0.001:
            return
        if abs(params.alpha[1] - self.alpha_m) > 0.001:
            return

        stats = tools.get_single_parameter_set_statistics(multi_run_stats)
        phi_n_index = numpy.argmin(numpy.abs(self.phi_n - params.phi[0]))
        phi_m_index = numpy.argmin(numpy.abs(self.phi_m - params.phi[1]))
        self.cov_of_variation[phi_n_index, phi_m_index] = stats.d_coeff_var
        self.f_collapse[phi_n_index, phi_m_index] = stats.f_collapse

    def __repr__(self):
        return f"_SimulationsForPoint(alpha_n={self.alpha_n}, alpha_m={self.alpha_m})"


def main():
    points = [
        # alpha_n, alpha_m
        _SimulationForPoint(alpha_n=0.2, alpha_m=-0.7),
        _SimulationForPoint(alpha_n=0.2, alpha_m=-0.2),
        _SimulationForPoint(alpha_n=0.95, alpha_m=-0.2),
        _SimulationForPoint(alpha_n=0.95, alpha_m=-0.7)
    ]

    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_fixed_D_differing_phi"):
        for point in points:
            point.offer_data_point(params, multi_run_stats)

    fig = plt.figure()
    axes = numpy.array(fig.subplots(nrows=2, ncols=2, sharex="all", sharey="all")).flatten()
    for ax, point in zip(axes, points):
        ax: Axes
        mappable = ax.imshow(point.f_collapse, interpolation="nearest",
                  cmap="gnuplot", vmin=0, vmax=1.3,
                  extent=(point.phi_m[0], point.phi_m[-1], point.phi_n[-1], point.phi_n[0]))
        if point.alpha_n > 0.9:
            ax.set_xlabel("phi_m")  # We're on the last row
        ax.set_ylabel("phi_n")
        ax.set_title(f"a_n={point.alpha_n}, a_m={point.alpha_m}")
        ax.invert_yaxis()
    plt.suptitle("Depletion (events/1000h)")
    plt.colorbar(mappable)
    plt.show()


if __name__ == '__main__':
    main()
