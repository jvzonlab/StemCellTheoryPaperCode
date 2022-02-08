import os.path
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable

import matplotlib.pyplot as plt
import numpy

from stem_cell_model import sweeper
from stem_cell_model.parameters import SimulationParameters
from stem_cell_model.results import MultiRunStats
from stem_cell_model.two_compartment_model import run_simulation

PARAMETER_SETS = [
    # alpha_n, alpha_m and phi
    [0.95, -0.95, 1.0],
    [0.2, -0.95, 1.0],
    [0.2, -0.2, 1.0],
    [0.2, -0.2, 0.25]
]
MAX_SIZE = 80

# Params for running in-place simulations
T = (16.153070175438597, 3.2357834505600382)  # Based on measured values
STEPS_ALONG_SIZE_AXIS = 60
T_SIM = int(1e5)  # Total simulation time


def _run_in_place_simulation(alpha_n: float, alpha_m: float, phi: float) -> Iterable[
    Tuple[SimulationParameters, MultiRunStats]]:
    """Runs a quick in-place simulation. Useful if the simulation data on disk is missing the requested paramters."""
    folder = f"two_comp_sweep_data_variable_S_alphan{alpha_n}_alpham{alpha_m}_phi{phi}"

    if not os.path.isdir(folder):
        # Run the simulation now
        print("Running in-place simulation for alpha_n, alpha_m, phi=", alpha_n, alpha_m, phi)

        S_values = set((10 ** numpy.linspace(0, 3, num=STEPS_ALONG_SIZE_AXIS, endpoint=True)).astype(numpy.int32))
        params_list = list()
        for S in S_values:
            params = SimulationParameters.for_S_alpha_and_phi(
                S=S, alpha_n=alpha_n, alpha_m=alpha_m, phi=phi, T=T, a=1 / T[0])
            if params is None:
                continue

            params_list.append(params)
        sweeper.sweep(run_simulation, params_list, output_folder=folder, t_sim=T_SIM)

    return sweeper.load_sweep_results(folder)


def _almost_equal(number1: float, number2: float) -> bool:
    return abs(number2 - number1) < 0.01


def _add_mean_and_variance(params: SimulationParameters, multi_run_stats: MultiRunStats, *,
                           means: Dict[int, List[float]], variances: Dict[int, List[float]]):
    parameter_set_index = None
    for i, parameter_set in enumerate(PARAMETER_SETS):
        if _almost_equal(params.alpha[0], parameter_set[0]) and _almost_equal(params.alpha[1],
                                                                              parameter_set[1]) and _almost_equal(
                params.phi[0], parameter_set[2]):
            parameter_set_index = i
    if parameter_set_index is None:
        return
    if params.D > MAX_SIZE:
        return

    mean = multi_run_stats.nm_mean / multi_run_stats.t_tot
    variance = multi_run_stats.nm_sq / multi_run_stats.t_tot - mean ** 2

    means[parameter_set_index].append(float(numpy.sum(mean)))
    variances[parameter_set_index].append(float(numpy.sum(variance)))

def main():
    fig = plt.figure()
    ax = fig.gca()

    means: Dict[int, List[float]] = defaultdict(list)
    variances: Dict[int, List[float]] = defaultdict(list)

    # Collect from earlier simulations
    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_variable_S"):
        _add_mean_and_variance(params, multi_run_stats, means=means, variances=variances)

    # Run in-place simulations for missing data
    for parameter_set_index in range(len(means)):
        if len(means[parameter_set_index]) > 0:
            continue  # Parameter set was part of the stored data

        alpha_n, alpha_m, phi = PARAMETER_SETS[parameter_set_index]
        for params, multi_run_stats in _run_in_place_simulation(alpha_n, alpha_m, phi):
            _add_mean_and_variance(params, multi_run_stats, means=means, variances=variances)

    # Sort
    for i in range(len(means)):
        variances[i] = [variance for _, variance in sorted(zip(means[i], variances[i]), key=lambda pair: pair[0])]
        means[i].sort()

    # Plot!
    for parameter_set_index, parameter_set in enumerate(PARAMETER_SETS):
        ax.plot(means[parameter_set_index], numpy.sqrt(variances[parameter_set_index]), label=str(parameter_set))
    ax.set_xlabel("Mean")
    ax.set_ylabel("Standard deviation")
    ax.legend()

    poisson_x_values = numpy.linspace(0, MAX_SIZE, MAX_SIZE * 5)
    poisson_y_values = numpy.sqrt(poisson_x_values)
    ax.plot(poisson_x_values, poisson_y_values, color="gray")

    ax.tick_params(direction="in")
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 25)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
