from typing import List

from matplotlib import pyplot as plt

from stem_cell_model import tools, sweeper
from stem_cell_model.parameters import SimulationParameters
from stem_cell_model.results import MultiRunStats
from stem_cell_model.tools import SingleParameterSetStatistics


class _SimulationForPoint:
    alpha_n: float
    alpha_m: float
    phi_n: float

    phi_m: List[float]
    results: List[SingleParameterSetStatistics]

    def __init__(self, *, alpha_n: float, alpha_m: float, phi_n: float):
        self.alpha_n = alpha_n
        self.alpha_m = alpha_m
        self.phi_n = phi_n

        self.phi_m = list()
        self.results = list()

    def offer_data_point(self, params: SimulationParameters, multi_run_stats: MultiRunStats):
        if abs(params.alpha[0] - self.alpha_n) > 0.001:
            return
        if abs(params.alpha[1] - self.alpha_m) > 0.001:
            return
        if abs(params.phi[0] - self.phi_n) > 0.001:
            return

        self.phi_m.append(params.phi[1])
        self.results.append(tools.get_single_parameter_set_statistics(multi_run_stats))

    def __repr__(self):
        return f"_SimulationsForPoint(alpha_n={self.alpha_n}, alpha_m={self.alpha_m}, phi_n={self.phi_n}," \
               f" phi_m={self.phi_m})"


def main():
    points = [
        # alpha_n, alpha_m, phi_n
        _SimulationForPoint(alpha_n=0.2, alpha_m=-0.95, phi_n=1),
        _SimulationForPoint(alpha_n=0.2, alpha_m=-0.2, phi_n=1),
        _SimulationForPoint(alpha_n=0.2, alpha_m=-0.2, phi_n=0.25),
        _SimulationForPoint(alpha_n=0.95, alpha_m=-0.95, phi_n=1)
    ]

    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_fixed_D_differing_phi"):
        for point in points:
            point.offer_data_point(params, multi_run_stats)

    fig = plt.figure()
    axes = fig.subplots(ncols=len(points), sharex=True, sharey=True)
    for ax, point in zip(axes, points):
        ax.plot(point.phi_m, [result.d_coeff_var for result in point.results])
    plt.show()


if __name__ == '__main__':
    main()
