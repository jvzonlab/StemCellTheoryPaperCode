"""In case you want to find out what makes your code so slow, run this file. It does a single run and prints what is taking up time."""

import cProfile

import numpy

from stem_cell_model.parameters import SimulationParameters, SimulationConfig
from stem_cell_model.results import MultiRunStats
from stem_cell_model.two_compartment_model import run_simulation


def main():
    T = (16.153070175438597, 3.2357834505600382)
    D = 20
    phi = 0.05
    params = SimulationParameters(n0=(D, 0), alpha=(0, 0), phi=(phi, phi), T=T, a=1/T[0], S=D)
    random = numpy.random.Generator(numpy.random.MT19937(seed=1))

    t_sim = 100000
    output = MultiRunStats()

    # run simulation
    while output.t_tot < t_sim:
        config = SimulationConfig(t_sim=t_sim - output.t_tot, random=random)
        res = run_simulation(config, params)
        output.add_results(res)

    # print run statistics
    output.print_run_statistics()


cProfile.run('main()')
