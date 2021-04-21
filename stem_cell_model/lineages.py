import math
from typing import List, Union, Tuple, Iterable, Dict, Any

import numpy
import numpy as np
import matplotlib.pyplot as plt
import operator

from stem_cell_model.division_counts import DivisionCounts


class CloneSizeDistribution:

    _clone_sizes: Dict[int, int]

    @staticmethod
    def of_single_clone(clone_size: int) -> "CloneSizeDistribution":
        """Returns a clone size "distribution" consisting of only a single clone."""
        distribution = CloneSizeDistribution()
        distribution._clone_sizes[clone_size] = 1
        return distribution

    @staticmethod
    def of_clone_sizes(*args: int)-> "CloneSizeDistribution":
        """Returns a clone size distribution of the given sizes. For example, (3, 4, 3, 5) is
        a clone size distribution where clone size 3 is the most frequent.."""
        distribution = CloneSizeDistribution()
        for clone_size in args:
            distribution.add_clone_size(clone_size)
        return distribution

    def __init__(self):
        self._clone_sizes = dict()

    def add_clone_size(self, clone_size: int):
        """Add a single clone size to this distribution."""
        if clone_size in self._clone_sizes:
            self._clone_sizes[clone_size] += 1
        else:
            self._clone_sizes[clone_size] = 1

    def merge(self, other: "CloneSizeDistribution"):
        """Adds all data from the other clone size distribution to this clone size distribution."""
        for clone_size, count in other._clone_sizes.items():
            if clone_size in self._clone_sizes:
                self._clone_sizes[clone_size] += count
            else:
                self._clone_sizes[clone_size] = count

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CloneSizeDistribution):
            return False
        return other._clone_sizes == self._clone_sizes

    def __repr__(self) -> str:
        return "CloneSizeDistribution(" + repr(self._clone_sizes) + ")"

    def max(self) -> int:
        """Gets the highest occuring clone size."""
        return max(self._clone_sizes.keys())

    def to_flat_array(self) -> numpy.ndarray:
        """Returns a flat array of the clone sizes. If this distribution has 5 occurrences of clone size
        3 and 2 occurrences of clone size 4, then this returns [3, 3, 3, 3, 3, 4, 4].

        Of course, this is quite wasteful for RAM. This method mainly exists for compatibility with
        old scripts that use matplotlib.pyplot.hist. You should instead use a bar plot with
        indices() and to_height_array()."""
        length = sum(self._clone_sizes.values())
        array = numpy.empty(length, dtype=numpy.uint16)

        i = 0
        for clone_size, count in self._clone_sizes.items():
            for _ in range(count):
                array[i] = clone_size
                i += 1
        return array

    def get_clone_size_count(self, clone_size: int) -> int:
        """Gets how many times the given clone size was found."""
        clone_size_count = self._clone_sizes.get(clone_size)
        if clone_size_count is None:
            return 0
        return clone_size_count

    def indices(self) -> List[int]:
        """Returns [1, 2, 3, ..., self.max()]."""
        return list(range(1, self.max() + 1))

    def to_height_array(self) -> List[int]:
        """Gets how often each clone size occurs, starting from clone size 1 (at position 0)."""
        return_values = list()
        for i in range(1, self.max() + 1):
            return_values.append(self.get_clone_size_count(i))
        return return_values

    def get_average_and_st_dev(self) -> Tuple[float, float]:
        total_size = sum(clone_size * count for clone_size, count in self._clone_sizes.items())
        total_count = sum(self._clone_sizes.values())
        average = total_size / total_count

        variance = 1 / (total_count - 1) * sum((clone_size ** 2) * count for clone_size, count in self._clone_sizes.items()) - (total_count / (total_count - 1)) * average ** 2

        return average, math.sqrt(variance)


class _LineageTrack:
    track_id: int
    track_start_time: int
    compartment: "_CompartmentByTime"

    # Stores whether this cell is proliferative. If this is True, it means that the cell will divide
    # (even if that division hasn't been simulated yet)
    is_proliferative: bool

    # The tracks of the daughters. Either empty or (daughter1, daughter2)
    daughters: Union[Tuple, Tuple["_LineageTrack", "_LineageTrack"]] = ()

    def __init__(self, track_id: int, track_start_time: int, compartment: int, is_proliferative: bool):
        self.track_id = track_id
        self.track_start_time = track_start_time
        self.compartment = _CompartmentByTime(compartment)
        self.is_proliferative = is_proliferative

    def get_clone_size(self, max_time: float) -> int:
        if self.track_start_time > max_time:
            raise ValueError("Track started after max_time")

        if len(self.daughters) == 2:
            daughter1, daughter2 = self.daughters
            if daughter1.track_start_time > max_time:
                return 1  # Don't include these daughters, the division happened after the time cutoff
            return daughter1.get_clone_size(max_time) + daughter2.get_clone_size(max_time)
        else:
            return 1

    def exists_at_time(self, time: float) -> bool:
        """Cells exist from start_time to (but not including) daughter.start_time. This function returns
        whether the given time falls in that range."""
        if time < self.track_start_time:
            return False
        if len(self.daughters) == 2 and time >= self.daughters[0].track_start_time:
            return False
        return True


class Lineage:

    # Id -> Track mapping. If multiple tracks share the same id
    # (which happens in a single lineage), the youngest is used. (This is the currently live cell.)
    _id_to_track: Dict[int, _LineageTrack]

    _tracks: List[_LineageTrack]

    n_cell: int

    def __init__(self, lin_id: int, lin_interval: int, lin_compartment: int, lin_is_dividing: bool, n_cell: int):
        self._id_to_track = dict()
        self._tracks = list()

        first_track = _LineageTrack(lin_id, lin_interval, lin_compartment, lin_is_dividing)
        self._id_to_track[first_track.track_id] = first_track
        self._tracks.append(first_track)

        self.n_cell=n_cell

    def divide_cell(self, id_mother, id_daughter_list, daughter_is_dividing_list, t_divide):
        # id_mother: label of mother cell
        # id_daughter_cell_list: list of [daughter1_id, daughter2_id]
        # daughter_is_dividing_list: List of [bool, bool], indicating whether a daughter continues dividing.
        # t_divide: time of division
        track = self._id_to_track.get(id_mother)
        if track is None:
            return

        self.n_cell += 1

        compartment = track.compartment.last_compartment()
        track_daughter_1 = _LineageTrack(id_daughter_list[0], t_divide, compartment, daughter_is_dividing_list[0])
        track_daughter_2 = _LineageTrack(id_daughter_list[1], t_divide, compartment, daughter_is_dividing_list[1])
        track.daughters = (track_daughter_1, track_daughter_2)

        self._tracks.append(track_daughter_1)
        self._tracks.append(track_daughter_2)

        self._id_to_track[track_daughter_1.track_id] = track_daughter_1
        self._id_to_track[track_daughter_2.track_id] = track_daughter_2

    def _get_sublineage_draw_data(self, track: _LineageTrack, t_end: int, x_curr_branch: float, x_end_branch: float, line_list):
    #def _get_sublineage_draw_data(self, lin_id, lin_interval, lin_compartment, t_end, x_curr_branch, x_end_branch, line_list):
        # if current branch doesn't have daughters
        if len(track.daughters) == 0:
            # then it has no sublineage, so we plot an end branch
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
                x_curr_branch, x_end_branch, line_list = self._get_sublineage_draw_data(daughter, t_end, x_curr_branch, x_end_branch, line_list)
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

    def get_lineage_draw_data(self, t_end):
        x_curr,x_end,line_list = self._get_sublineage_draw_data(self._tracks[0], t_end, 0, 0, [])
        return x_end, line_list

    def draw_lineage(self,T_end,x_offset,show_cell_id=False,col_comp_0='r', col_default='k'):
        (diagram_width, line_list)=self.get_lineage_draw_data(T_end)
    
        for l in line_list:
            X=l[0]
            T=l[1]
            CID=l[2]
            comp=l[3]
            col = col_comp_0 if comp == 0 else col_default
            if len(T)==2:
                ## two timepoints T, so this is a vertical line
                # plot line
                plt.plot( [x_offset+X[0],x_offset+X[0]], T, linestyle='-', color=col )
                if show_cell_id:
                    # print cell id
                    plt.text( x_offset+X[0], T[0], CID)
            if len(X)==2:
                ## two x positions, so this a horizontal line indicating division
                # plot line
                plt.plot( [x_offset+X[0],x_offset+X[1]], [T[0],T[0]], linestyle='-', color=col )
        return(diagram_width)

    # checks if a cell with id=cell_id is in this lineage        
    def is_cell_in_lineage(self, cell_id):
        return cell_id in self._id_to_track

    # gets the clone size distribution of this lineage tree. For each cell that exists at min_time, the clone size
    # at max_time is returned.
    def get_clone_size_distribution(self, min_time: float, max_time: float) -> CloneSizeDistribution:
        distribution = CloneSizeDistribution()
        for track in self._tracks:
            if track.exists_at_time(min_time):
                distribution.add_clone_size(track.get_clone_size(max_time))
        return distribution

    # gets the clone size distributions of the given duration for this lineage tree.
    # If min_time is 0, max_time is 70, duration is 50 and increment is 5, then this will return the clone sizes for
    # [0, 50], [5, 55], [10, 60], [15, 65] and [20, 70.
    def get_clone_size_distributions_with_duration(self, min_time: float, max_time: float, duration: float, increment: int = 5) -> CloneSizeDistribution:
        clone_sizes = CloneSizeDistribution()
        for start_time in range(int(min_time), int(max_time - duration + 1), increment):
            clone_sizes.merge(self.get_clone_size_distribution(start_time, start_time + duration))
        return clone_sizes

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
            if switch_time > current_time:
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


def _is_single_number(value):
    value_type = type(value)
    return value_type == float or value_type == np.float64 or value_type == int or value_type == np.int


def _is_single_boolean(value):
    return value is True or value is False
