"""Like two_compartment_model_sweep, but with space and aT = 1."""
import numpy

from stem_cell_model import sweeper
from stem_cell_model.parameters import SimulationParameters
from stem_cell_model.two_compartment_model_space import run_simulation_niche


def run_simulation(steps_along_axis: int = 40):
    # Arguments
    # sweep alpha for stem cell compartment from 0.1 to 1.0
    alpha_n_values = numpy.linspace(25, 1000, steps_along_axis) * .001
    # sweep alpha for transit amplifying compartment from -1.0 to -0.1
    alpha_m_values = numpy.linspace(-1000, -25, steps_along_axis) * .001
    # sweep degree of symmetry phi from 0.1 to 1.0
    phi_values = numpy.linspace(25, 1000, steps_along_axis) * .001

    T = (16.153070175438597, 3.2357834505600382)  # Based on measured values
    t_sim = int(1e5)  # Total simulation time
    D = 30  # average number of dividing cells
    n_max = 5 * D  # Maximum number of dividing cells
    output_folder = "two_comp_sweep_data_fixed_D_aT1"

    # Build all possible parameters
    params_list = list()
    for alpha_m in alpha_m_values:
        for phi in phi_values:
            for alpha_n in alpha_n_values:
                params = SimulationParameters.for_D_alpha_and_phi(
                    D=D, alpha_n=alpha_n, alpha_m=alpha_m, phi=phi, T=T, a=1/T[0], n_max=n_max)
                if params is not None:
                    params_list.append(params)

    # Go!
    sweeper.sweep(run_simulation_niche, params_list, t_sim=t_sim, output_folder=output_folder)


if __name__ == '__main__':
    # Wrapping it like this is necessary to use the multiprocessing module in sweeper.sweep
    run_simulation()
