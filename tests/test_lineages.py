from unittest import TestCase

from stem_cell_model import clone_size_distributions
from stem_cell_model.clone_size_distributions import CloneSizeDistribution
from stem_cell_model.lineages import Lineage


class TestLineages(TestCase):

    def test_clone_sizes(self):
        """Cell 1 divides into [2, 3], and cell 2 into [4, 5] while cell 3 stops dividing."""
        lineages = Lineage(lin_id=1, lin_interval=1, lin_compartment=0, lin_is_dividing=False, n_cell=1)
        lineages.divide_cell(id_mother=1, id_daughter_list=[2, 3], daughter_is_dividing_list=[True, False], t_divide=10)
        lineages.divide_cell(id_mother=2, id_daughter_list=[4, 5], daughter_is_dividing_list=[False, False],
                             t_divide=20)

        self.assertEqual(CloneSizeDistribution.of_single_clone(3),
                         clone_size_distributions.get_clone_size_distribution(lineages, 2, 25))  # One clone of size three

        # If we start after the first division, we should see one clone of size 2 and one clone of size 1
        self.assertEqual(CloneSizeDistribution.of_clone_sizes(2, 1),
                         clone_size_distributions.get_clone_size_distribution(lineages, 11, 25))

    def test_is_cell_in_lineage(self):
        """Cell 1 divides into [2, 3], and cell 2 into [4, 5] while cell 3 stops dividing."""
        lineages = Lineage(lin_id=1, lin_interval=1, lin_compartment=0, lin_is_dividing=False, n_cell=1)
        lineages.divide_cell(id_mother=1, id_daughter_list=[2, 3], daughter_is_dividing_list=[True, False], t_divide=10)
        lineages.divide_cell(id_mother=2, id_daughter_list=[4, 5], daughter_is_dividing_list=[False, False],
                             t_divide=20)

        self.assertTrue(lineages.is_cell_in_lineage(4))
        self.assertFalse(lineages.is_cell_in_lineage(6))

    def test_count_divisions(self):
        """Cell 1 divides into [2, 3], and cell 2 into [4, 5] while cell 3 stops dividing."""
        lineages = Lineage(lin_id=1, lin_interval=1, lin_compartment=0, lin_is_dividing=False, n_cell=1)
        lineages.divide_cell(id_mother=1, id_daughter_list=[2, 3], daughter_is_dividing_list=[True, False], t_divide=10)
        lineages.divide_cell(id_mother=2, id_daughter_list=[4, 5], daughter_is_dividing_list=[False, False],
                             t_divide=20)

        division_counts = lineages.count_divisions()
        self.assertEqual(1, division_counts.sisters_symmetric_non_dividing)
        self.assertEqual(1, division_counts.sisters_asymmetric)
        self.assertEqual(0, division_counts.sisters_symmetric_dividing)

