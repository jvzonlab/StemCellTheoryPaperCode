"""Specifically tests the CloneSizeDistribution class."""
from unittest import TestCase

import numpy

from stem_cell_model.clone_size_distributions import CloneSizeDistribution


class TestCloneSizes(TestCase):

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
