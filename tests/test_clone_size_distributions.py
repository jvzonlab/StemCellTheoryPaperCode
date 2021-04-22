"""Specifically tests the CloneSizeDistribution class."""
from unittest import TestCase

import numpy

from stem_cell_model import clone_size_distributions, timed_clone_size_distributions
from stem_cell_model.clone_size_distributions import CloneSizeDistribution
from stem_cell_model.lineages import Lineages, LineageTrack


class TestCloneSizes(TestCase):

    def test_clone_size_distribution(self):
        """Cell 1 divides into [2, 3], and cell 2 into [4, 5] while cell 3 stops dividing."""
        lineages = Lineages()
        lineages.add_lineage(lin_id=1, lin_interval=1, lin_compartment=0, lin_is_dividing=False)
        lineages.divide_cell(id_mother=1, id_daughter_list=[2, 3], daughter_is_dividing_list=[True, False], t_divide=10)
        lineages.divide_cell(id_mother=2, id_daughter_list=[4, 5], daughter_is_dividing_list=[False, False],
                             t_divide=20)

        self.assertEqual(CloneSizeDistribution.of_single_clone(3),
                         clone_size_distributions.get_clone_size_distribution(lineages, 2, 25))  # One clone of size three

        # If we start after the first division, we should see one clone of size 2 and one clone of size 1
        self.assertEqual(CloneSizeDistribution.of_clone_sizes(2, 1),
                         clone_size_distributions.get_clone_size_distribution(lineages, 11, 25))

    def test_clone_size_distribution_proliferative_only(self):
        """Cell 1 divides into [2, 3], which both divide and then all stop dividing."""
        lineages = Lineages()
        lineages.add_lineage(lin_id=1, lin_interval=1, lin_compartment=0, lin_is_dividing=False)
        lineages.divide_cell(id_mother=1, id_daughter_list=[2, 3], daughter_is_dividing_list=[True, True], t_divide=10)
        lineages.divide_cell(id_mother=2, id_daughter_list=[4, 5], daughter_is_dividing_list=[False, False],
                             t_divide=20)
        lineages.divide_cell(id_mother=3, id_daughter_list=[6, 7], daughter_is_dividing_list=[False, False],
                             t_divide=20)

        timed_distribution = timed_clone_size_distributions.get_proliferative_clone_size_distribution(lineages, 1, 25, interval=8)

        self.assertEqual(CloneSizeDistribution.of_clone_sizes(0), timed_distribution.last())

    def test_array(self):
        clone_size = CloneSizeDistribution()
        clone_size.add_clone_size(2)
        clone_size.add_clone_size(2)
        clone_size.add_clone_size(2)
        clone_size.add_clone_size(4)
        clone_size.add_clone_size(4)
        self.assertEqual([2, 2, 2, 4, 4], list(clone_size.to_flat_array()))

    def test_statistics(self):
        clone_size = CloneSizeDistribution()
        clone_size.add_clone_size(2)
        clone_size.add_clone_size(2)
        clone_size.add_clone_size(2)
        clone_size.add_clone_size(4)
        clone_size.add_clone_size(4)

        mean, std = clone_size.get_average_and_st_dev()
        self.assertAlmostEqual(float(numpy.mean(clone_size.to_flat_array())), mean)
        self.assertAlmostEqual(float(numpy.std(clone_size.to_flat_array(), ddof=1)), std)
