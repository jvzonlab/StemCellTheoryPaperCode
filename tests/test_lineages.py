from unittest import TestCase

from stem_cell_model.lineages import Lineages, LineageTrack


class TestLineages(TestCase):

    def test_clone_size_proliferative(self):
        track = LineageTrack(track_id=1, track_start_time=0, compartment=0, is_proliferative=True)

        daughter1 = LineageTrack(track_id=2, track_start_time=10, compartment=0, is_proliferative=True)
        daughter2 = LineageTrack(track_id=3, track_start_time=10, compartment=0, is_proliferative=True)
        track.daughters = (daughter1, daughter2)

        granddaughter1_1 = LineageTrack(track_id=4, track_start_time=20, compartment=0, is_proliferative=False)
        granddaughter1_2 = LineageTrack(track_id=5, track_start_time=20, compartment=0, is_proliferative=False)
        daughter1.daughters = (granddaughter1_1, granddaughter1_2)

        granddaughter2_1 = LineageTrack(track_id=6, track_start_time=20, compartment=0, is_proliferative=False)
        granddaughter2_2 = LineageTrack(track_id=7, track_start_time=20, compartment=0, is_proliferative=False)
        daughter2.daughters = (granddaughter2_1, granddaughter2_2)

        self.assertEqual(1, track.get_clone_size(9))
        self.assertEqual(2, track.get_clone_size(19))
        self.assertEqual(4, track.get_clone_size(29))

        self.assertEqual(1, track.get_proliferative_niche_clone_size(9))
        self.assertEqual(2, track.get_proliferative_niche_clone_size(19))
        self.assertEqual(0, track.get_proliferative_niche_clone_size(29))

    def test_clone_size_niche(self):
        track = LineageTrack(track_id=1, track_start_time=0, compartment=0, is_proliferative=True)

        # Daughter 2 moves out of the niche at T=15
        daughter1 = LineageTrack(track_id=2, track_start_time=10, compartment=0, is_proliferative=True)
        daughter2 = LineageTrack(track_id=3, track_start_time=10, compartment=0, is_proliferative=True)
        daughter2.compartment.add_move(15, towards_compartment=1)
        track.daughters = (daughter1, daughter2)

        granddaughter1_1 = LineageTrack(track_id=4, track_start_time=20, compartment=0, is_proliferative=False)
        granddaughter1_2 = LineageTrack(track_id=5, track_start_time=20, compartment=0, is_proliferative=False)
        granddaughter1_2.compartment.add_move(30, towards_compartment=1)
        granddaughter1_1.compartment.add_move(40, towards_compartment=1)
        daughter1.daughters = (granddaughter1_1, granddaughter1_2)

        granddaughter2_1 = LineageTrack(track_id=6, track_start_time=20, compartment=1, is_proliferative=False)
        granddaughter2_2 = LineageTrack(track_id=7, track_start_time=20, compartment=1, is_proliferative=False)
        daughter2.daughters = (granddaughter2_1, granddaughter2_2)

        self.assertEqual(1, track.get_niche_clone_size(9))
        self.assertEqual(2, track.get_niche_clone_size(14))
        self.assertEqual(1, track.get_niche_clone_size(15))  # Here one cell moved out
        self.assertEqual(2, track.get_niche_clone_size(29))  # Here the other cell in the niche divided
        self.assertEqual(1, track.get_niche_clone_size(35))  # Here one of those daughters moved away
        self.assertEqual(0, track.get_niche_clone_size(45))  # Here the last daughter in the niche moved away

    def test_is_cell_in_lineage(self):
        """Cell 1 divides into [2, 3], and cell 2 into [4, 5] while cell 3 stops dividing."""
        lineages = Lineages()
        lineages.add_lineage(lin_id=1, lin_interval=1, lin_compartment=0, lin_is_dividing=False)
        lineages.divide_cell(id_mother=1, id_daughter_list=[2, 3], daughter_is_dividing_list=[True, False], t_divide=10)
        lineages.divide_cell(id_mother=2, id_daughter_list=[4, 5], daughter_is_dividing_list=[False, False],
                             t_divide=20)

        self.assertTrue(lineages.is_cell_in_lineage(4))
        self.assertFalse(lineages.is_cell_in_lineage(6))

    def test_count_divisions(self):
        """Cell 1 divides into [2, 3], and cell 2 into [4, 5] while cell 3 stops dividing."""
        lineages = Lineages()
        lineages.add_lineage(lin_id=1, lin_interval=1, lin_compartment=0, lin_is_dividing=False)
        lineages.divide_cell(id_mother=1, id_daughter_list=[2, 3], daughter_is_dividing_list=[True, False], t_divide=10)
        lineages.divide_cell(id_mother=2, id_daughter_list=[4, 5], daughter_is_dividing_list=[False, False],
                             t_divide=20)

        division_counts = lineages.count_divisions()
        self.assertEqual(1, division_counts.sisters_symmetric_non_dividing)
        self.assertEqual(1, division_counts.sisters_asymmetric)
        self.assertEqual(0, division_counts.sisters_symmetric_dividing)

