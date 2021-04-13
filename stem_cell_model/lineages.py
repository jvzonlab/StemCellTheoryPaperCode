from typing import List, Union, Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt
import operator

from stem_cell_model.division_counts import DivisionCounts


class Lineage:

    lin_id: Union[List, int]
    lin_interval: Union[List, int]
    lin_compartment: Union[List, "_CompartmentByTime"]
    lin_is_dividing: Union[List, bool]  # Stores whether the cells is going to divide. This is only useful for cells that have not yet divided
    n_cell: int

    def __init__(self, lin_id: int, lin_interval: int, lin_compartment: int, lin_is_dividing: bool, n_cell: int):
        self.lin_id=lin_id
        self.lin_interval=lin_interval
        self.lin_compartment=_CompartmentByTime(lin_compartment)
        self.lin_is_dividing=lin_is_dividing
        self.n_cell=n_cell

    def lineage_divide_cell(self,lin_id, lin_interval, lin_compartment, lin_is_dividing, t_div, ind_mother, ind_daughter_list, daughter_is_dividing_list, n_cell):
        # if current branch is not a list
        if type(lin_id)!=list:
            # then it has not sublineage
            if lin_id==ind_mother:
                # if the id of this branch is that of the dividing cell, implement the division
                # replace the id with a list of the two daughter cell ids
                lin_id=[ind_mother,ind_daughter_list]
                # replace the time of birth of cell <lin_id> with an interval
                # [t_birth_<lin_id>, [t_birth_daughter0, t_birth_daughter1]]
                lin_interval=[lin_interval,[t_div,t_div]]
                # replace the compartment with copies of the same compartment
                lin_compartment=[lin_compartment, [
                    _CompartmentByTime(lin_compartment.last_compartment()),
                    _CompartmentByTime(lin_compartment.last_compartment())
                ]]
                # replace the lin_is
                lin_is_dividing=[lin_is_dividing, daughter_is_dividing_list]
                # and increase the cell count for the lineage
                n_cell=n_cell+1
        else:
            # if it is a list, it has a sublineage
            for i in range(0,2):
                # for each daughter sublineage, get the id and time interval data
                sub_lin_id = lin_id[1][i]
                sub_lin_interval = lin_interval[1][i]
                sub_lin_compartment = lin_compartment[1][i]
                sub_lin_is_dividing = lin_is_dividing[1][i]
                # and search for cell (and implement division when found) in sublineage
                (sub_lin_id,sub_lin_interval,sub_lin_compartment,sub_lin_is_dividing,n_cell) = self.lineage_divide_cell(
                        sub_lin_id, sub_lin_interval, sub_lin_compartment, sub_lin_is_dividing, t_div, ind_mother, ind_daughter_list, daughter_is_dividing_list, n_cell)
                # and update sublineages in lineage data
                lin_id[1][i]=sub_lin_id
                lin_interval[1][i]=sub_lin_interval
                lin_compartment[1][i]=sub_lin_compartment
                lin_is_dividing[1][i]=sub_lin_is_dividing
    
        # return updated lineage data        
        return (lin_id,lin_interval,lin_compartment,lin_is_dividing,n_cell)

    def divide_cell(self, id_mother, id_daughter_list, daughter_is_dividing_list, t_divide):
        # id_mother: label of mother cell
        # id_daughter_cell_list: list of [daughter1_id, daughter2_id]
        # daughter_is_dividing_list: List of [bool, bool], indicating whether a daughter continues dividing.
        # t_divide: time of division
        (self.lin_id,self.lin_interval,self.lin_compartment,self.lin_is_dividing,self.n_cell)=\
                self.lineage_divide_cell(self.lin_id,self.lin_interval,self.lin_compartment,self.lin_is_dividing,t_divide,id_mother,id_daughter_list,daughter_is_dividing_list,self.n_cell)

    def get_sublineage_draw_data(self,lin_id,lin_interval,lin_compartment,t_end,x_curr_branch,x_end_branch,line_list):
        # if current branch is not a list
        if type(lin_id)!=list:
            # then it has no sublineage, so we plot an end branch
            # set x position of current branch to that of the next end branch
            x_curr_branch=x_end_branch
            # plot line from time of birth to end time of lineage tree
#            plt.plot([x_curr_branch,x_curr_branch],[lin_interval,t_end],'-k')
            X=[x_curr_branch]
            T=[lin_interval,t_end]
            CID=lin_id
            for T, comp in lin_compartment.get_all_compartments_with_times(T):
                line_list.append( [X,T,CID,comp] )
#            plt.text(x_curr_branch, lin_interval, lin_id)
            # and increase the position of the next end branch 
            x_end_branch=x_end_branch+1
        else:
            # if it is a list, it has a sublineage
            x=[]
            for i in range(0,2):
                # for each daughter sublineage, get the id and time interval data
                sub_lin_id = lin_id[1][i]
                sub_lin_interval = lin_interval[1][i]
                sub_lin_compartment = lin_compartment[1][i]
                # and draw sublineage sublineage
                (x_curr_branch,x_end_branch,line_list) = self.get_sublineage_draw_data(sub_lin_id, sub_lin_interval,
                         sub_lin_compartment, t_end,x_curr_branch,x_end_branch,line_list)
                # for each sublineage, save the current branch x position
                x.append(x_curr_branch)
            # get the start of the time interval
            t0=lin_interval[0]
            CID=lin_id[0]
            compartments=lin_compartment[0]
            # and the end            
            if type(lin_interval[1][0])!=list:
                t1=lin_interval[1][0]
#                cell_id=lin_id[1][0]
            else:
                t1=lin_interval[1][0][0]
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
        return (x_curr_branch,x_end_branch,line_list)

    def get_lineage_draw_data(self, t_end):
        (x_curr,x_end,line_list)=self.get_sublineage_draw_data(self.lin_id,self.lin_interval,self.lin_compartment,t_end,0,0,[])
        return( (x_end,line_list) )
        
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
        
#    def draw_lineage(self,lin_id, lin_interval,t_end,x_curr_branch,x_end_branch):
#        # if current branch is not a list
#        if type(lin_id)!=list:
#            # then it has not sublineage, so we plot an end branch
#            # set x position of current branch to that of the next end branch
#            x_curr_branch=x_end_branch
#            # plot line from time of birth to end time of lineage tree
#            plt.plot([x_curr_branch,x_curr_branch],[lin_interval,t_end],'-k')
#            # and increase the position of the next end branch 
#            x_end_branch=x_end_branch+1
#        else:
#            # if it is a list, it has a sublineage
#            x=[]
#            for i in range(0,2):
#                # for each daughter sublineage, get the id and time interval data
#                sub_lin_id = lin_id[1][i]
#                sub_lin_interval = lin_interval[1][i]
#                # and draw sublineage sublineage
#                (x_curr_branch,x_end_branch) = self.draw_lineage(sub_lin_id, sub_lin_interval,t_end,x_curr_branch,x_end_branch)
#                # for each sublineage, save the current branch x position
#                x.append(x_curr_branch)
#            # get the start of the time interval
#            t0=lin_interval[0]
#            # and the end            
#            if type(lin_interval[1][0])!=list:
#                t1=lin_interval[1][0]
#            else:
#                t1=lin_interval[1][0][0]
#            # plot horizontal line connected the two daughter branches
#            plt.plot([x[0],x[1]], [ t1,t1 ], '-k')
#            # and plot the mother branch
#            x_curr_branch=(x[0]+x[1])/2.
#            plt.plot([x_curr_branch,x_curr_branch], [ t0,t1 ], '-k')
#    
#        # return updated lineage data        
#        return (x_curr_branch,x_end_branch)

    def is_in_lineage(self,lin_id, cell_id):
        if type(lin_id)!=list:
            if lin_id==cell_id:
                return(True)
        else:
            if lin_id[0]==cell_id:
                return(True)
            elif type(lin_id[1])==list:
                for i in range(0,2):
                    if self.is_in_lineage(lin_id[1][i], cell_id):
                        return(True)
            elif lin_id[1]==cell_id:
                    return(True)
                
        return(False)

    # checks if a cell with id=cell_id is in this lineage        
    def is_cell_in_lineage(self, cell_id):
        return ( self.is_in_lineage(self.lin_id, cell_id) )

    # gets the clone size distribution of this lineage tree. For each cell that exists at min_time, the clone size
    # at max_time is returned.
    def get_clone_size_distribution(self, min_time: float, max_time: float) -> List[int]:
        return self._get_sub_clone_size_distribution(self.lin_interval, min_time, max_time, "")

    # gets the clone size distributions of the given duration for this lineage tree.
    # If min_time is 0, max_time is 70, duration is 50 and increment is 5, then this will return the clone sizes for
    # [0, 50], [5, 55], [10, 60], [15, 65] and [20, 70.
    def get_clone_size_distributions_with_duration(self, min_time: float, max_time: float, duration: float, increment: int = 5) -> List[int]:
        clone_sizes = []
        for start_time in range(int(min_time), int(max_time - duration + 1), increment):
            clone_sizes += self.get_clone_size_distribution(start_time, start_time + duration)
        return clone_sizes

    # gets the clone size distribution for the given sub-lineage. For each cell that exists at min_time, the clone size
    # at max_time is returned.
    def _get_sub_clone_size_distribution(self, lin_interval: Union[float, List], min_time: float, max_time: float, indent: str) -> List[int]:
        if _is_single_number(lin_interval):
            # this is a non-dividing cell starting at lin_interval
            time_start = lin_interval
            division_time = None
            daughter1, daughter2 = None, None
        else:
            # we have a division
            time_start, next = lin_interval
            daughter1, daughter2 = next
            division_time = daughter1 if _is_single_number(daughter1) else daughter1[0]

        if time_start > max_time:
            return []  # cell didn't exist yet at this time point - report no clone size

        if division_time is None or division_time > min_time:
            # cell continues to exist (without dividing) until min_time is reached
            return [self._get_clone_size(lin_interval, max_time)]

        # need to search deeper in the lineage
        clone_sizes = []
        clone_sizes += self._get_sub_clone_size_distribution(daughter1, min_time, max_time, indent + "│ ")
        clone_sizes += self._get_sub_clone_size_distribution(daughter2, min_time, max_time, indent + "│ ")
        return clone_sizes

    # returns the clone size of the given sub-lineage, ignoring any divisions happening after max_time
    def _get_clone_size(self, lin_interval: Union[float, List], max_time: float) -> int:
        if _is_single_number(lin_interval):
            return 1

        # we have a division, lin_interval structure is [time_start, [daughter1, daughter2]]
        time_start, next = lin_interval
        daughter1, daughter2 = next

        # calculate division time from time_start of an arbitrary daughter
        division_time = daughter1 if _is_single_number(daughter1) else daughter1[0]
        if division_time > max_time:
            # division happens after our time window, act as if this cell hasn't divided yet
            return 1

        return self._get_clone_size(daughter1, max_time) + self._get_clone_size(daughter2, max_time)

    def move_cell(self, cell_id: int, t: int, towards_component: int):
        # moves the cell at the given time point to the given compartment
        (self.lin_id, self.lin_interval, self.lin_compartment) = self._lineage_move_cell(
                self.lin_id, self.lin_interval, self.lin_compartment, cell_id, t, towards_component)

    def _lineage_move_cell(self, lin_id, lin_interval, lin_compartment, moving_cell_id, time_move, towards_compartment):
        # if current branch is not a list
        if type(lin_id) != list:
            # then it has not sublineage
            if lin_id == moving_cell_id:
                # if the id of this branch is that of the moving cell, implement the movement
                lin_compartment.add_move(time_move, towards_compartment)
        else:
            # if it is a list, it has a sublineage
            daughter1 = lin_interval[1][1]
            time_division = daughter1 if isinstance(daughter1, float) else daughter1[0]
            if time_move < time_division:
                if lin_id[0] == moving_cell_id:
                    # implement move in parent
                    lin_compartment[0].add_move(time_move, towards_compartment)
            else:
                # implement move in daughter
                for i in range(0, 2):
                    # for each daughter sublineage, get the id and time interval data
                    sub_lin_id = lin_id[1][i]
                    sub_lin_interval = lin_interval[1][i]
                    sub_lin_compartment = lin_compartment[1][i]
                    # and search for cell (and implement division when found) in sublineage
                    (sub_lin_id, sub_lin_interval, sub_lin_compartment) = self._lineage_move_cell(
                        sub_lin_id, sub_lin_interval, sub_lin_compartment, moving_cell_id, time_move, towards_compartment)
                    # and update sublineages in lineage data
                    lin_id[1][i] = sub_lin_id
                    lin_interval[1][i] = sub_lin_interval
                    lin_compartment[1][i] = sub_lin_compartment

        # return updated lineage data
        return (lin_id, lin_interval, lin_compartment)

    def count_divisions(self) -> DivisionCounts:
        counter = DivisionCounts()
        self._count_divisions_0(self.lin_is_dividing, counter)
        return counter

    def _count_divisions_0(self, lin_is_dividing, counter: DivisionCounts):
        if _is_single_boolean(lin_is_dividing):
            # End of line
            return
        else:
            # Found a division
            sister1, sister2 = lin_is_dividing[1]
            sister1_dividing = bool(sister1) if _is_single_boolean(sister1) else sister1[0]
            sister2_dividing = bool(sister2) if _is_single_boolean(sister2) else sister2[0]
            counter.add_sister_entry(sister1_dividing, sister2_dividing)

            # Look at granddaughters for cousin analysis
            if not _is_single_boolean(sister1) and not _is_single_boolean(sister2):
                # We can look at the granddaughters
                for cousin1 in sister1[1]:
                    for cousin2 in sister2[1]:
                        cousin1_dividing = bool(cousin1) if _is_single_boolean(cousin1) else cousin1[0]
                        cousin2_dividing = bool(cousin2) if _is_single_boolean(cousin2) else cousin2[0]
                        counter.add_cousin_entry(cousin1_dividing, cousin2_dividing)
            
            # Continue the search down this rabbit hole
            self._count_divisions_0(sister1, counter)
            self._count_divisions_0(sister2, counter)


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
