"""In case you want to find out what makes your code so slow, run this file. It does a single run and prints what is taking up time."""

import cProfile

import numpy

from stem_cell_model.parameters import SimulationParameters, SimulationConfig
from stem_cell_model.two_compartment_model_space import run_sim_niche, run_simulation_niche


def main():
    T = (16.153070175438597, 3.2357834505600382)
    params = SimulationParameters.for_S_alpha_and_phi(S=252, alpha_n=0.1, alpha_m=-0.4, phi=0.675, T=T, a=100/T[0])
    random = numpy.random.Generator(numpy.random.MT19937(seed=1))
    config = SimulationConfig(t_sim=10000, n_max=5*30, random=random)
    res = run_simulation_niche(config, params)


cProfile.run('main()')
