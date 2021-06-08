import math
from typing import List, Union, Tuple, Iterable, Dict, Any

import numpy
import numpy as np
import matplotlib.pyplot as plt
import operator

from matplotlib.axes import Axes

from stem_cell_model.division_counts import DivisionCounts


class LineageTrack:
    track_id: int
    track_start_time: int
    compartment: "_CompartmentByTime"

    # Stores whether this cell is proliferative. If this is True, it means that the cell will divide
    # (even if that division hasn't been simulated yet)
    is_proliferative: bool

    # The tracks of the daughters. Either empty or (daughter1, daughter2)
    daughters: Union[Tuple, Tuple["LineageTrack", "LineageTrack"]] = ()

    def __init__(self, track_id: int, track_start_time: int, compartment: int, is_proliferative: bool):
        self.track_id = track_id
        self.track_start_time = track_start_time
        self.compartment = _CompartmentByTime(compartment)
        self.is_proliferative = is_proliferative

    def get_clone_size(self, max_time: float) -> int:
        """Gets how many cells this cell will eventually produce. For example, if this cell divides, and both daughters
        divide too, this method returns 4."""
        if self.track_start_time > max_time:
            raise ValueError("Track started after max_time")

        if len(self.daughters) == 2:
            daughter1, daughter2 = self.daughters
            if daughter1.track_start_time > max_time:
                return 1  # Don't include these daughters, the division happened after the time cutoff
            return daughter1.get_clone_size(max_time) + daughter2.get_clone_size(max_time)
        else:
            return 1

    def get_proliferative_clone_size(self, max_time: float) -> int:
        """Gets how many dividing cells this track will eventually produce. If this track divides into two dividing
        daughter cells, and one of those daughters divides into two dividing cells, then the clone size is three.

        Returns 0 if this cell doesn't divide. Returns 1 if this track is proliferative, but hasn't divided yet."""
        if self.track_start_time > max_time:
            raise ValueError("Track started after max_time")

        if len(self.daughters) == 2:
            daughter1, daughter2 = self.daughters
            if daughter1.track_start_time > max_time:
                return 1 if self.is_proliferative else 0  # Don't include these daughters, the division happened after the time cutoff
            return daughter1.get_proliferative_clone_size(max_time) + daughter2.get_proliferative_clone_size(max_time)
        else:
            return 1 if self.is_proliferative else 0

    def get_niche_clone_size(self, max_time: float) -> int:
        """Gets how many tracks in the niche compartment this track will eventually produce. Returns 0 if all cells
        leave the niche."""
        if self.track_start_time > max_time:
            raise ValueError("Track started after max_time")

        if len(self.daughters) == 2:
            daughter1, daughter2 = self.daughters
            if daughter1.track_start_time <= max_time:
                return daughter1.get_niche_clone_size(max_time) + daughter2.get_niche_clone_size(max_time)

        # Assume no division
        if self.compartment.get_compartment_at(max_time) == 0:
            return 1
        return 0  # Cell didn't survive in the niche compartment

    def exists_at_time(self, time: float) -> bool:
        """Cells exist from start_time to (but not including) daughter.start_time. This function returns
        whether the given time falls in that range."""
        if time < self.track_start_time:
            return False
        if len(self.daughters) == 2 and time >= self.daughters[0].track_start_time:
            return False
        return True


class Lineages:

    # Id -> Track mapping. If multiple tracks share the same id
    # (which happens in a single lineage), the youngest is used. (This is the currently live cell.)
    _id_to_track: Dict[int, LineageTrack]

    _tracks: List[LineageTrack]
    _lineage_starts: List[LineageTrack]

    n_cell: int

    def __init__(self):
        self._id_to_track = dict()
        self._tracks = list()
        self._lineage_starts = list()
        self.n_cell = 0

    def add_lineage(self, lin_id: int, lin_interval: int, lin_compartment: int, lin_is_dividing: bool):
        """Starts a new lineage. Raises ValueError if the given id is already in use."""
        if lin_id in self._id_to_track:
            raise ValueError("Duplicate lineage id: " + str(lin_id))
        first_track = LineageTrack(lin_id, lin_interval, lin_compartment, lin_is_dividing)
        self._id_to_track[first_track.track_id] = first_track
        self._tracks.append(first_track)
        self._lineage_starts.append(first_track)
        self.n_cell += 1

    def __len__(self) -> int:
        """Gets the number of lineage starts."""
        return len(self._lineage_starts)

    def __getitem__(self, item) -> LineageTrack:
        """Gets a particular lineage start"""
        return self._lineage_starts[item]

    def divide_cell(self, id_mother: int, id_daughter_list: Union[List[int], Tuple[int, int]],
                    daughter_is_dividing_list: Union[List[bool], Tuple[bool, bool]], t_divide: int):
        # id_mother: label of mother cell
        # id_daughter_cell_list: list of [daughter1_id, daughter2_id]
        # daughter_is_dividing_list: List of [bool, bool], indicating whether a daughter continues dividing.
        # t_divide: time of division
        track = self._id_to_track.get(id_mother)
        if track is None:
            return

        self.n_cell += 1

        compartment = track.compartment.last_compartment()
        track_daughter_1 = LineageTrack(id_daughter_list[0], t_divide, compartment, daughter_is_dividing_list[0])
        track_daughter_2 = LineageTrack(id_daughter_list[1], t_divide, compartment, daughter_is_dividing_list[1])
        track.daughters = (track_daughter_1, track_daughter_2)

        self._tracks.append(track_daughter_1)
        self._tracks.append(track_daughter_2)

        self._id_to_track[track_daughter_1.track_id] = track_daughter_1
        self._id_to_track[track_daughter_2.track_id] = track_daughter_2

    def draw_lineages(self, ax: Axes, t_end: int, x_offset: int = 0, show_cell_id=False, col_comp_0='r', col_default='k'):
        """Draws the lineage tree of all lineages."""
        for track in self._lineage_starts:
            diagram_width = _draw_single_lineage(ax, track, t_end, x_offset, show_cell_id, col_comp_0, col_default)
            x_offset += diagram_width

    def draw_single_lineage(self, ax: Axes, track: LineageTrack, t_end: int, x_offset, show_cell_id=False,
                            col_comp_0='r', col_default='k') -> int:
        """Draws the lineage tree of a single lineage. Returns the width of the lineage tree."""
        return _draw_single_lineage(ax, track, t_end, x_offset, show_cell_id, col_comp_0, col_default)

    # checks if a cell with id=cell_id is in this lineage
    def is_cell_in_lineage(self, cell_id):
        return cell_id in self._id_to_track

    def move_cell(self, cell_id: int, t: int, towards_component: int):
        # moves the cell at the given time point to the given compartment
        track = self._id_to_track.get(cell_id)
        if track is None:
            return
        track.compartment.add_move(t, towards_component)

    def count_divisions(self) -> DivisionCounts:
        counter = DivisionCounts()
        for track in self._tracks:
            # Check all tracks

            if len(track.daughters) != 2:
                continue  # No division to check
            sister1, sister2 = track.daughters[0], track.daughters[1]
            counter.add_sister_entry(sister1.is_proliferative, sister2.is_proliferative)

            if len(sister1.daughters) == 2 and len(sister2.daughters) == 2:
                # We can check the cousins
                for cousin1 in sister1.daughters:
                    for cousin2 in sister2.daughters:
                        counter.add_cousin_entry(cousin1.is_proliferative, cousin2.is_proliferative)

        return counter

    def get_tracks(self) -> Iterable[LineageTrack]:
        """Gets all tracks in the lineage. A track is a single vertical line in the lineage tree."""
        yield from self._tracks


# Lineage drawing
def _get_lineage_draw_data(track: LineageTrack, t_end: int):
    x_curr, x_end, line_list = _get_sublineage_draw_data(track, t_end, 0, 0, [])
    return x_end, line_list

def _get_sublineage_draw_data(track: LineageTrack, t_end: int, x_curr_branch: float, x_end_branch: float, line_list):
    # if current branch doesn't have daughters
    if len(track.daughters) == 0 or track.daughters[0].track_start_time >= t_end:
        # then it has no sublineage (at least not within the displayed time), so we plot an end branch
        # set x position of current branch to that of the next end branch
        x_curr_branch=x_end_branch
        # plot line from time of birth to end time of lineage tree
#            plt.plot([x_curr_branch,x_curr_branch],[lin_interval,t_end],'-k')
        X=[x_curr_branch]
        T=[track.track_start_time,t_end]
        CID=track.track_id
        for T, comp in track.compartment.get_all_compartments_with_times(T):
            line_list.append( [X,T,CID,comp] )
#            plt.text(x_curr_branch, track.track_start_time, track.track_id)
        # and increase the position of the next end branch
        x_end_branch=x_end_branch+1
    else:
        # if has a sublineage
        x=[]
        for i in range(0,2):
            # for each daughter sublineage, get the id and time interval data
            daughter = track.daughters[i]
            # and draw sublineage sublineage
            x_curr_branch, x_end_branch, line_list = _get_sublineage_draw_data(daughter, t_end, x_curr_branch, x_end_branch, line_list)
            # for each sublineage, save the current branch x position
            x.append(x_curr_branch)
        # get the start of the time interval
        t0 = track.track_start_time
        CID = track.track_id
        compartments = track.compartment
        # and the end
        t1 = track.daughters[0].track_start_time
        # plot horizontal line connected the two daughter branches
#            plt.plot([x[0],x[1]], [ t1,t1 ], '-k')
        X=[x[0],x[1]]
        T=[t1]
        line_list.append( [X,T,CID,compartments.last_compartment()] )

        # and plot the mother branch
        x_curr_branch=(x[0]+x[1])/2.
#            plt.plot([x_curr_branch,x_curr_branch], [ t0,t1 ], '-k')
#            plt.text(x_curr_branch, t0, cell_id)
        X=[x_curr_branch]
        T=[t0,t1]
        for T, comp in compartments.get_all_compartments_with_times(T):
            line_list.append( [X,T,CID,comp] )

    # return updated lineage data
    return x_curr_branch, x_end_branch, line_list


def _draw_single_lineage(ax: Axes, track: LineageTrack, t_end: int, x_offset, show_cell_id=False, col_comp_0='r',
                         col_default='k'):
    (diagram_width, line_list) = _get_lineage_draw_data(track, t_end)

    for l in line_list:
        X = l[0]
        T = l[1]
        CID = l[2]
        comp = l[3]
        col = col_comp_0 if comp == 0 else col_default
        if len(T) == 2:
            ## two timepoints T, so this is a vertical line
            # plot line
            ax.plot([x_offset + X[0], x_offset + X[0]], T, linestyle='-', color=col)
            if show_cell_id:
                # print cell id
                ax.text(x_offset + X[0], T[0], CID)
        if len(X) == 2:
            ## two x positions, so this a horizontal line indicating division
            # plot line
            ax.plot([x_offset + X[0], x_offset + X[1]], [T[0], T[0]], linestyle='-', color=col)
    return diagram_width


class _CompartmentByTime:
    # we can only store one variable per track. To still keep track of at which time point the cell moved to another
    # compartment, we use this class, which can store different compartments for different time points

    _starting_compartment: int
    _moves: List[Tuple[int, int]]  # list of (time point, new compartment)

    def __init__(self, starting_compartment: int):
        self._starting_compartment = starting_compartment
        self._moves = []

    def get_all_compartments_with_times(self, T: Tuple[int, int]) -> Iterable[Tuple[Tuple[int, int], int]]:
        # yields pairs of (T, compartment) for all time frames with a different compartment. T is [min_time, max_time].
        if len(self._moves) == 0:
            # no moves, so the result is trivial
            yield T, self._starting_compartment

        start_time, end_time = T
        current_time = start_time
        current_compartment = self._starting_compartment

        for switch_time, new_compartment in self._moves:
            if switch_time >= current_time:
                yield [current_time, switch_time], current_compartment  # finish off current compartment
                # and start new one
                current_time = switch_time
                current_compartment = new_compartment

        # finish off last compartment
        if end_time > current_time:
            yield [current_time, end_time], current_compartment


    def add_move(self, time: int, towards_compartment: int):
        self._moves.append((time, towards_compartment))
        self._moves.sort(key=operator.itemgetter(0))  # sort by the first element of the tuple, which is the time

    def last_compartment(self) -> int:
        # gets the compartment after all moves are done
        if len(self._moves) == 0:
            return self._starting_compartment
        return self._moves[-1][1]

    def __repr__(self) -> str:
        if len(self._moves) > 0:
            return f"<_CompartmentByTime({self._starting_compartment}) with moves>"
        return f"_CompartmentByTime({self._starting_compartment})"

    def get_compartment_at(self, time: float) -> int:
        if len(self._moves) == 0:
            return self._starting_compartment

        previous_compartment = self._starting_compartment
        for move_time, move_compartment in self._moves:
            if time < move_time:
                return previous_compartment
            previous_compartment = move_compartment
        return self.last_compartment()



def _is_single_number(value):
    value_type = type(value)
    return value_type == float or value_type == np.float64 or value_type == int or value_type == np.int


def _is_single_boolean(value):
    return value is True or value is False
