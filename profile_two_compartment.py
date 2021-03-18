"""In case you want to find out what makes your code so slow, run this file. It does a single run and prints what is taking up time."""

import cProfile

import numpy

from stem_cell_model.two_compartment_model_space import run_sim_niche


def main():
    numpy.random.seed(50)
    params = {'S': 300, 'alpha': [0, 0], 'phi': [0.9, 0.9], 'T': [16, 3], 'a': 10}
    res = run_sim_niche(5000, 100000, params, n0=[100, 10])


cProfile.run('main()')
