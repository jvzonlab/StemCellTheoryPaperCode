"""Plots the one compartment model, which in practise is implemented by setting the parameters of
both compartments the same. D and phi are varied, alpha is 0 to keep homeostasis. S doesn't matter since
both comparments are the same, so for simplicity D == S."""
import numpy

from stem_cell_model import sweeper
from stem_cell_model.parameters import SimulationParameters
from stem_cell_model.two_compartment_model import run_simulation


def main(steps_along_axis: int = 40):
    # Arguments

    # sweep degree of symmetry phi from 0 to 1
    phi_values = numpy.linspace(0, 1, steps_along_axis)
    # sweep average number of cells from 1 to 100 (logarithmic)
    D_values = set((10 ** numpy.linspace(0, 2, num=steps_along_axis, endpoint=True)).astype(numpy.int32))

    T = (16.153070175438597, 3.2357834505600382)  # Based on measured values
    t_sim = 1000  # Total simulation time
    output_folder = "one_comp_sweep_data_var_D_var_phi"

    # Build all possible parameters
    params_list = list()
    for D in D_values:
        for phi in phi_values:
            params = SimulationParameters(n0=(D, 0), alpha=(0, 0), phi=(phi, phi), T=T, a=1/T[0], S=D)
            if params is not None:
                params_list.append(params)

    # Go!
    sweeper.sweep(run_simulation, params_list, t_sim=t_sim, n_max=100000, output_folder=output_folder)


if __name__ == '__main__':
    # Wrapping it like this is necessary to use the multiprocessing module in sweeper.sweep
    main()
