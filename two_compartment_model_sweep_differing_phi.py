"""Like two_compartment_model_sweep, but with space and aT = 1."""
import numpy

from stem_cell_model import sweeper, two_compartment_model
from stem_cell_model.parameters import SimulationParameters


def run_simulation(steps_along_axis: int = 40):
    points = [
        # alpha_n, alpha_m
        [0.2, -0.7],
        [0.2, -0.2],
        [0.95, -0.2],
        [0.95, -0.7]
    ]

    # Arguments
    # sweep degree of symmetry phi from 0.025 to 1.0
    phi_n_values = numpy.linspace(25, 1000, steps_along_axis) * .001
    phi_m_values = numpy.linspace(25, 1000, steps_along_axis) * .001

    T = (16.153070175438597, 3.2357834505600382)  # Based on measured values
    t_sim = int(1e5)  # Total simulation time
    D = 30  # average number of dividing cells
    n_max = 5 * D  # Maximum number of dividing cells
    output_folder = "two_comp_sweep_data_fixed_D_differing_phi"

    # Build all possible parameters
    params_list = list()
    for alpha_n, alpha_m in points:
        for phi_n in phi_n_values:
            for phi_m in phi_m_values:
                params = SimulationParameters.for_D_alpha_and_phi_ext(
                    D=D, alpha_n=alpha_n, alpha_m=alpha_m, phi_n=phi_n, phi_m=phi_m, T=T, n_max=n_max)
                if params is not None:
                    params_list.append(params)

    # Go!
    sweeper.sweep(two_compartment_model.run_simulation, params_list, t_sim=t_sim, output_folder=output_folder)


if __name__ == '__main__':
    # Wrapping it like this is necessary to use the multiprocessing module in sweeper.sweep
    run_simulation()
