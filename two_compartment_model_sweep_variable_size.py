"""Runs a simulation for different values of S (compartment size), phi (symmetry) and alpha_n = -alpha_n."""
import numpy

from stem_cell_model import sweeper
from stem_cell_model.parameters import SimulationParameters
from stem_cell_model.two_compartment_model_space import run_simulation_niche


def main(steps_along_alpha_and_phi_axis: int = 40, steps_along_S_axis: int = 60):
    # Arguments
    S_values = set((10 ** numpy.linspace(0, 3, num=steps_along_S_axis, endpoint=True)).astype(numpy.int32))
    phi_values = numpy.linspace(0.025, 1, num=steps_along_alpha_and_phi_axis, endpoint=True)
    alpha_n_values = numpy.linspace(0.025, 1, num=steps_along_alpha_and_phi_axis, endpoint=True)
    T = (16.153070175438597, 3.2357834505600382)  # Based on measured values
    t_sim = int(1e5)  # Total simulation time
    output_folder = "two_comp_sweep_data_variable_S_aT1"

    # Build all possible parameters
    params_list = list()
    for S in S_values:
        for phi in phi_values:
            for alpha_n in alpha_n_values:
                params = SimulationParameters.for_S_alpha_and_phi(
                    S=S, alpha_n=alpha_n, alpha_m=-alpha_n, phi=phi, T=T, a=1/T[0])
                if params is not None:
                    params_list.append(params)

    # Go!
    sweeper.sweep(run_simulation_niche, params_list, t_sim=t_sim, output_folder=output_folder)


if __name__ == '__main__':
    # Wrapping it like this is necessary to use the multiprocessing module in sweeper.sweep
    main()
