"""Like two_compartment_model_sweep_space, but with aT = 100."""
import numpy

from stem_cell_model import sweeper
from stem_cell_model.parameters import SimulationParameters
from stem_cell_model.two_compartment_model_space import run_simulation_niche


def main(steps_along_axis: int = 40):
    # Arguments
    T = (16.153070175438597, 3.2357834505600382)  # Based on measured values

    # sweep rearrangement from 0.001/T[0] to 10000/T[0] on a log scale
    a_values = 10 ** numpy.linspace(-3, 4, steps_along_axis) / T[0]

    t_sim = int(1e5)  # Total simulation time
    D = 30  # average number of dividing cells
    alpha_n = 0.95
    alpha_m = -0.95
    phi = 0.95
    output_folder = "two_comp_sweep_data_variable_mixing"

    # Build all possible parameters
    params_list = list()
    for a in a_values:
        params = SimulationParameters.for_D_alpha_and_phi(
            D=D, alpha_n=alpha_n, alpha_m=alpha_m, phi=phi, T=T, a=a)
        if params is not None:
            params_list.append(params)

    # Go!
    sweeper.sweep(run_simulation_niche, params_list, t_sim=t_sim, output_folder=output_folder)


if __name__ == '__main__':
    # Wrapping it like this is necessary to use the multiprocessing module in sweeper.sweep
    main()
