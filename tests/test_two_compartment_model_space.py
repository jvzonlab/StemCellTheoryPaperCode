from unittest import TestCase

import numpy

from stem_cell_model.two_compartment_model_space import run_sim_niche


class TestTwoCompartmentModelSpace(TestCase):

    def test_run_sim_niche(self):
        """Just a general check to make sure that the simulation runs"""
        numpy.random.seed(1234)  # Fixed seed to keep simulation the same
        params = {'S': 30, 'alpha': [0, 0], 'phi': [0.9, 0.9], 'T': [16, 3], 'a': 1/16}
        res = run_sim_niche(50, 100000, params, n0=[10, 10], track_n_vs_t=True,
                            track_lineage_time_interval=[0, 100])
        self.assertFalse(res["RunStats"]["run_ended_early"])
        self.assertFalse(res["RunStats"]["n_exploded"])

    def test_run_sim_niche_dead(self):
        """Lets the niche die on purpose by setting the growth rate to -0.5."""
        numpy.random.seed(1234)  # Fixed seed to keep simulation the same
        params = {'S': 30, 'alpha': [-0.5, -0.5], 'phi': [1, 1], 'T': [16, 3], 'a': 1/16}
        res = run_sim_niche(100, 100000, params, n0=[10, 10], track_n_vs_t=True,
                            track_lineage_time_interval=[0, 100])
        self.assertTrue(res["RunStats"]["run_ended_early"])
        self.assertFalse(res["RunStats"]["n_exploded"])
