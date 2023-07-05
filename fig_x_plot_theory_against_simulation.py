import math

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from stem_cell_model import sweeper, tools


def plot_alpha_n(ax: Axes):
    alpha_n_values = list()
    theoretical_D_values = list()
    simulated_D_values = list()
    simulated_D_values_stdev = list()

    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_variable_S"):
        if params.S == 29 and params.alpha[0] + params.alpha[1] < 0.00001 and params.phi[0] == 1:
            alpha_n, alpha_m = params.alpha
            D = math.log(1 + alpha_n) * params.S * (alpha_m - alpha_n) / alpha_m
            statistics = tools.get_single_parameter_set_statistics(multi_run_stats)

            alpha_n_values.append(alpha_n)
            theoretical_D_values.append(D)
            simulated_D_values.append(statistics.d_mean)
            simulated_D_values_stdev.append(statistics.d_std)

    sorting = numpy.argsort(alpha_n_values)
    alpha_n_values = numpy.array(alpha_n_values)[sorting]
    theoretical_D_values = numpy.array(theoretical_D_values)[sorting]
    simulated_D_values = numpy.array(simulated_D_values)[sorting]
    simulated_D_values_stdev = numpy.array(simulated_D_values_stdev)[sorting]

    ax.plot(alpha_n_values, simulated_D_values, color="#6a01e0", label="Simulations")
    ax.plot(alpha_n_values, theoretical_D_values, color="#fae100", label="Equation")
    ax.fill_between(alpha_n_values, simulated_D_values - simulated_D_values_stdev,
                    simulated_D_values + simulated_D_values_stdev, color="#6a01e0", alpha=0.1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("alpha_n, -alpha_m")
    ax.set_ylabel("D")
    ax.legend()


def plot_S(ax: Axes):
    S_values = list()
    theoretical_D_values = list()
    simulated_D_values = list()
    simulated_D_values_stdev = list()

    for params, multi_run_stats in sweeper.load_sweep_results("two_comp_sweep_data_variable_S"):
        if abs(params.alpha[0] - 0.5) < 0.00001 and abs(params.alpha[1] + 0.5) < 0.00001\
                and params.phi[0] == 1:
            alpha_n, alpha_m = params.alpha
            D = math.log(1 + alpha_n) * params.S * (alpha_m - alpha_n) / alpha_m
            statistics = tools.get_single_parameter_set_statistics(multi_run_stats)

            S_values.append(params.S)
            theoretical_D_values.append(D)
            simulated_D_values.append(statistics.d_mean)
            simulated_D_values_stdev.append(statistics.d_std)

    sorting = numpy.argsort(S_values)
    S_values = numpy.array(S_values)[sorting]
    theoretical_D_values = numpy.array(theoretical_D_values)[sorting]
    simulated_D_values = numpy.array(simulated_D_values)[sorting]
    simulated_D_values_stdev = numpy.array(simulated_D_values_stdev)[sorting]

    ax.plot(S_values, simulated_D_values, color="#6a01e0", label="Simulations")
    ax.plot(S_values, theoretical_D_values, color="#fae100", label="Equation")
    ax.fill_between(S_values, simulated_D_values - simulated_D_values_stdev,
                    simulated_D_values + simulated_D_values_stdev, color="#6a01e0", alpha=0.1)
    ax.set_xlabel("S")
    ax.set_ylabel("D")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()


def main():
    figure = plt.figure(figsize=(5.5,3))
    ax_a, ax_b = figure.subplots(nrows=1, ncols=2)
    plot_S(ax_a)
    plot_alpha_n(ax_b)
    figure.tight_layout()
    figure.suptitle("phi=1")
    plt.show()


if __name__ == "__main__":
    main()
