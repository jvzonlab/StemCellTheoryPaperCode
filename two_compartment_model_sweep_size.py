"""Runs a simulation for different values of S (compartment size), phi (symmetry) and alpha_n = -alpha_n."""
import numpy

from stem_cell_model import sweeper
from stem_cell_model.parameters import SimulationParameters
from stem_cell_model.two_compartment_model_space import run_simulation_niche


def run_simulation():
    # Arguments
    S_values = set((10 ** numpy.linspace(0, 2, num=40, endpoint=True)).astype(numpy.int32))
    phi_values = numpy.linspace(0.025, 1, num=40, endpoint=True)
    alpha_n_values = numpy.linspace(0.025, 1, num=40, endpoint=True)
    T = (16.153070175438597, 3.2357834505600382)  # Based on measured values
    t_sim = int(1e5)  # Total simulation time
    n_max = 1000000  # Maximum number of dividing cells, will never be reached with this value
    output_folder = "two_comp_sweep_data_variable_S"

    # Build all possible parameters
    params_list = list()
    for S in S_values:
        for phi in phi_values:
            for alpha_n in alpha_n_values:
                params = SimulationParameters.for_S_alpha_and_phi(
                    S=S, alpha_n=alpha_n, alpha_m=-alpha_n, phi=phi, T=T)
                if params is not None:
                    params_list.append(params)

    # Go!
    sweeper.sweep(run_simulation_niche, params_list, t_sim=t_sim, n_max=n_max, output_folder=output_folder)


if __name__ == '__main__':
    # Wrapping it like this is necessary to use the multiprocessing module in sweeper.sweep
    run_simulation()
