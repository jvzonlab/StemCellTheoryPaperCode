from typing import List, Iterable

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray

from stem_cell_model import tools, sweeper
from stem_cell_model.parameters import SimulationParameters
from stem_cell_model.results import MultiRunStats


class _SimulationForPoint:
    alpha_n: float
    alpha_m: float

    phi_n: ndarray  # Linearly increasing phi_n
    phi_m: ndarray  # Linearly increasing phi_m
    cov_of_variation_d: ndarray  # Indexed as phi_n, phi_m
    d_mean: ndarray  # Indexed as phi_h, phi_m
    f_collapse: ndarray  # Indexed as phi_h, phi_m

    def __init__(self, *, alpha_n: float, alpha_m: float, steps_along_axis: int = 40):
        self.phi_n = numpy.linspace(25, 1000, steps_along_axis) * 0.001
        self.phi_m = numpy.linspace(25, 1000, steps_along_axis) * 0.001
        self.cov_of_variation_d = numpy.full((self.phi_n.shape[0], self.phi_m.shape[0]), numpy.nan)
        self.f_collapse = numpy.copy(self.cov_of_variation_d)
        self.d_mean = numpy.copy(self.cov_of_variation_d)

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
        self.cov_of_variation_d[phi_n_index, phi_m_index] = stats.d_coeff_var
        self.f_collapse[phi_n_index, phi_m_index] = stats.f_collapse
        self.d_mean[phi_n_index, phi_m_index] = stats.d_mean

    def __repr__(self):
        return f"_SimulationsForPoint(alpha_n={self.alpha_n}, alpha_m={self.alpha_m})"


def _find_point(points: Iterable[_SimulationForPoint], alpha_n: float, alpha_m: float) -> _SimulationForPoint:
    for point in points:
        if point.alpha_n == alpha_n and point.alpha_m == alpha_m:
            return point
    raise ValueError(f"Point with alpha_n={alpha_n} and alpha_m={alpha_m} not found")


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

    #_plot_d_mean(points)
    _plot_cov_of_variation_d(points)


def _plot_d_mean(points: List[_SimulationForPoint]):
    fig = plt.figure()
    axes = fig.subplots(nrows=2, ncols=2, sharex="all", sharey="all")
    for ax, point in zip(numpy.array(axes).flatten(), points):
        ax: Axes
        mappable = ax.imshow(point.d_mean, interpolation="nearest",
                             cmap="gnuplot", vmin=20, vmax=50,
                             extent=(point.phi_m[0], point.phi_m[-1], point.phi_n[-1], point.phi_n[0]))
        if point.alpha_n > 0.9:
            ax.set_xlabel("phi_m")  # We're on the last row
        ax.set_ylabel("phi_n")
        ax.set_title(f"a_n={point.alpha_n}, a_m={point.alpha_m}")
        ax.invert_yaxis()
    plt.suptitle("<D(t)>")
    plt.colorbar(mappable, ax=axes[:, 1], shrink=0.6)
    plt.show()


def _plot_cov_of_variation_d(points: List[_SimulationForPoint]):
    fig = plt.figure()
    axes = fig.subplots(nrows=3, ncols=2, sharex="all")
    for ax, point in zip(numpy.array(axes).flatten()[0:4], points):
        ax: Axes
        mappable = ax.imshow(point.cov_of_variation_d, interpolation="nearest",
                             cmap="gnuplot", vmin=0, vmax=0.6,
                             extent=(point.phi_m[0], point.phi_m[-1], point.phi_n[-1], point.phi_n[0]))
        if point.alpha_n > 0.9:
            ax.set_xlabel("phi_m")  # We're on the last row
        ax.set_ylabel("phi_n")
        ax.set_title(f"a_n={point.alpha_n}, a_m={point.alpha_m}")
        ax.invert_yaxis()

    ax_bottom_left = axes[2, 0]
    point = _find_point(points, alpha_n=0.2, alpha_m=-0.2)
    ax_bottom_left.plot(point.phi_n, point.cov_of_variation_d[:, numpy.searchsorted(point.phi_m, 0.25)],
                        label="alpha_n=0.2, alpha_m=-0.2, phi_m=0.25")
    ax_bottom_left.plot(point.phi_n, point.cov_of_variation_d[:, numpy.searchsorted(point.phi_m, 0.95)],
                        label="alpha_n=0.2, alpha_m=-0.2, phi_m=0.95")
    point = _find_point(points, alpha_n=0.95, alpha_m=-0.7)
    ax_bottom_left.plot(point.phi_n, point.cov_of_variation_d[:, numpy.searchsorted(point.phi_m, 0.95)],
                         label="alpha_n=0.95, alpha_m=-0.7, phi_m=0.95")

    ax_bottom_left.set_ylim(0, 0.6)
    ax_bottom_left.set_aspect("equal")
    ax_bottom_left.set_ylabel("CoV in D")

    ax_bottom_right: Axes = axes[2, 1]
    point = _find_point(points, alpha_n=0.2, alpha_m=-0.2)
    ax_bottom_right.plot(point.phi_m, point.cov_of_variation_d[numpy.searchsorted(point.phi_n, 0.25)], label="alpha_n=0.2, alpha_m=-0.2, phi_n=0.25")
    ax_bottom_right.plot(point.phi_m, point.cov_of_variation_d[numpy.searchsorted(point.phi_n, 0.95)], label="alpha_n=0.2, alpha_m=-0.2, phi_n=0.95")

    point = _find_point(points, alpha_n=0.95, alpha_m=-0.7)
    ax_bottom_right.plot(point.phi_m, point.cov_of_variation_d[numpy.searchsorted(point.phi_n, 0.95)], label="alpha_n=0.95, alpha_m=-0.7, phi_n=0.95")

    point = _find_point(points, alpha_n=0.95, alpha_m=-0.2)
    ax_bottom_right.plot(point.phi_m, point.cov_of_variation_d[numpy.searchsorted(point.phi_n, 0.95)], label="alpha_n=0.95, alpha_m=-0.2, phi_n=0.95")

    ax_bottom_right.set_ylim(0, 0.6)
    ax_bottom_right.set_aspect("equal")
    ax_bottom_right.set_ylabel("CoV in D")
    ax_bottom_right.legend()

    plt.suptitle("Coefficient of variation")
    plt.colorbar(mappable, ax=axes[:, 1], shrink=0.6)
    plt.show()


if __name__ == '__main__':
    main()
