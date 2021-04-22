
import numpy as np
from numpy import ndarray
from numpy.random import Generator

from stem_cell_model import parameters
from stem_cell_model.lineages import Lineages
from stem_cell_model.parameters import SimulationConfig, SimulationParameters
from stem_cell_model.results import MomentData, SimulationResults, RunStats
from stem_cell_model.two_compartment_model import Cell, get_next_dividing, DivisionTimer


# implement cell reorderings in niche.
# params: a - # reorderings / time / cell, t - time interval during which reorderings occur
def reorder_niche(random: Generator, niche: ndarray, a: float, t: int):
    # get niche size <S>
    S = len(niche)
    if S <= 1:
        return niche  # nothing to reorder
    # get the number of reorderings N in time interval <t>
    # this is given by a possion distribution with reordering rate a*S
    N = random.poisson(a * S * t)

    # get <N> random positions in niche where cells are reordered
    indices = random.integers(low=0, high=S - 1, size=N, endpoint=False)
    for x in indices:
        # swap that cell with the one above
        cell_one_above = niche[x + 1]
        niche[x + 1] = niche[x]
        niche[x] = cell_one_above

    # return new niche
    return niche


def run_sim_niche( t_sim,n_max, params, n0=[0,0], track_lineage_time_interval=[], track_n_vs_t=False):
    """
    Old method to run the niche simulation - please use the one below for new code.

    :param t_sim: Total simulation time
    :param n_max:
    :param params: dict containing cell cycle time <T>, stem cell compartment size <S>,
                   degrees of symmetry <phi_i> and growth rate <alpha_i> for compartment <i>
    :param n0: initial number of stem cells in compartment <i>
    :param track_lineage_time_interval: list [t_start, t_end] during which lineage information
                                        should be recorded. If empty, no lineage recorderd
    :param track_n_vs_t: track cell number versus time. If <false> only calculate moments
    :return: Simulation result object as a raw dictionary.
    """
    return run_simulation_niche(*parameters.from_old_format(
        t_sim, n_max, params, n0, track_lineage_time_interval, track_n_vs_t)).to_dict()


def run_simulation_niche(config: SimulationConfig, params: SimulationParameters) -> SimulationResults:
    """Run the simulation where the niche is a 1D column. When a cell in the niche divides, the
    uppermost cell in the niche is moved to the next compartment."""
    random = config.random
    division_timer = DivisionTimer(random)

    # if an interval to track lineages is defined
    if config.track_lineage_time_interval is not None:
        # then set flag for tracking them
        track_lineage=True
    else:
        track_lineage=False

    # initialize niche - array of cell ids (both dividing and non-dividing) in order
    niche = np.zeros(params.S, dtype=int)

    # initialize current cell id
    cell_id=1
    cell_list=[]  # List of dividing cells
    # initialize dividing cells in stem cell compartment
    for n in range(0, params.n0[0]):
        # assign parameters
        age = params.T[0] * random.random()
        # add cell
        cell_list.append(Cell(cell_id, 0, age, division_timer))
        # place cell in niche
        inserted_cell=False
        while not inserted_cell:
            ind = int(random.random() * params.S)
            if niche[ind] == 0:
                inserted_cell = True
                niche[ind] = cell_id
        # increase id
        cell_id += 1
    # initialize dividing cells outside compartment
    for n in range(0, params.n0[1]):
        age = params.T[0] * random.random()
        cell_list.append(Cell(cell_id, 1, age, division_timer))
        cell_id += 1

    ### calculate p,q parameters for both compartment

    p=[]
    q=[]
    for compartment in [0,1]:
        p.append((params.phi[compartment] + params.alpha[compartment]) / 2)
        q.append((params.phi[compartment] - params.alpha[compartment]) / 2)

    ### init for data collection

    # initalize simulation time
    t=0
    # intialize stem cell number n(t) = [ N(t), M(t) ], where N(t) is # stem cells
    # in compartment 1 and M(t) is # of stem cells in compartment 2
    n = np.array(params.n0, dtype=int)
    # initialize differentiated cell number u_i(t) in compartment <i>
    u = np.array([params.S - params.n0[0], 0], dtype=int)

    # initialize moment data for fluctuations in stem cell number
    moment_data = MomentData()

    # run stats
    run_ended_early=False
    t_end=0
    n_events=0

    if config.track_n_vs_t:
        # initialize tracks of stem cell numbers N(t) and M(t)
        n_vs_t = [ [t, n[0], n[1]] ]
        # and same for numbers of differentiated cells
        u_vs_t = [ [t, u[0], u[1]] ]

    # if an interval to track lineages is defined
    if config.track_lineage_time_interval is not None:
        # then set flag for tracking them
        track_lineage = True
        # initialize lineage list
        lineages = Lineages()
    else:
        track_lineage = False

    ### run simulation

    tracking_lineage=False
    cont=True
    while cont:
        # get time dt to next division
        mother_cell_index, dt = get_next_dividing(cell_list)

        # if time of division is before end of simulation
        if t + dt < config.t_sim:
            # implement the next division

            # print (n,u)

            # set new time
            t=t+dt
            # print("--- t=%f ---" % t)

            # adjust moments
            moment_data.adjust_moment_data(dt,n)

            # add dt to age
            for cell in cell_list:
                cell.time_to_division -= dt

            # implement cell reorderings for the intervening time dt
            niche = reorder_niche(random, niche, params.a, dt)

            # get compartment of dividing cell
            compartment=cell_list[mother_cell_index].comp

            ### get type of division

            # draw random number in (0,1)
            r = random.random()
            if r<=p[compartment]:
                # if r in (0,p), then div -> div + div
                div_type=0
            elif r<=(p[compartment]+q[compartment]):
                # if r in (p,p+q), then div -> non-div + non-div
                div_type=2
            else:
                # else, div -> div + non-div
                div_type=1

            daughter_cell_id_list=[]
            daughter_is_dividing_list=[]
            mother_cell_id = cell_list[mother_cell_index].id

            ### execute division
            if div_type==0:
                # div -> div + div
                # generate two new dividing cells to compartment <c>
                for i in [0,1]:
                    # add new cells to cell list
                    cell_list.append(Cell(cell_id, compartment, 0, division_timer))
                    # if needed, remember daughter cell id
                    daughter_cell_id_list.append( cell_id )
                    daughter_is_dividing_list.append( True )
                    # adjust cell id
                    cell_id += 1
                # adjust number of dividing cells
                n[compartment] += 1

            elif div_type==1:
                # div -> div + non-div
                # add a single dividing cell to compartment <c>
                cell_list.append(Cell(cell_id, compartment, 0, division_timer))
                # if tracking_lineage:
                # rember id of this daughter, if tracking lineage
                daughter_cell_id_list.append( cell_id )
                daughter_is_dividing_list.append( True )
                cell_id += 1
                # if tracking_lineage:
                # and remember id of the non-dividing daughter
                daughter_cell_id_list.append( cell_id )
                daughter_is_dividing_list.append( False )
                cell_id += 1
                # adjust number of non-dividing differentiated cells
                u[compartment] += 1

            elif div_type==2:
                # div -> non-div + non-div
                for i in [0,1]:
                    # if tracking_lineage:
                    # remember daughter cell ids
                    daughter_cell_id_list.append( cell_id )
                    daughter_is_dividing_list.append( False )
                # and increase cell id
                    cell_id += 1
                # adjust number of dividing cells
                n[compartment] -= 1
                # adjust number of non-dividing differentiated cells
                u[compartment] += 2

            # remove old cell after division
            del cell_list[mother_cell_index]

            # if division was in niche/compartment 0
            if compartment==0:
                # get position <x> in niche of mother cell
                x = [x for (x,y) in enumerate(niche) if y == mother_cell_id][0]
                # at that position insert two daughter cells
                if daughter_is_dividing_list[0] == True:
                    # put the id there in case it is a dividing cell
                    niche[ x ] = daughter_cell_id_list[0]
                else:
                    # or just zero if non-dividing
                    niche[ x ] = 0
                if daughter_is_dividing_list[1] == True:
                    # same for the other daughter
                    niche = np.insert(niche,x+1,daughter_cell_id_list[1])
                else:
                    niche = np.insert(niche,x+1,0)
                # get id of cell moved to compartment 1, it is the last cell in the niche
                cell_id_remove = niche[-1]
                # remove it from the niche
                niche = np.delete(niche,-1)
                if cell_id_remove == 0:
                    # adjust number of non-dividing differentiated cells
                    u[0] -= 1
                    u[1] += 1
                else:
                    # set compartment of removed cell to 1
                    ind = [x for (x,y) in enumerate(cell_list) if y.id == cell_id_remove][0]
                    cell_list[ ind ].comp=1
                                        # implement cell moving in saved lineages
                    if tracking_lineage:
                        lineages.move_cell(cell_list[ ind ].id, t, cell_list[ ind ].comp)

                    # adjust number of dividing stem cells
                    n[0] -= 1
                    n[1] += 1

            # save number of dividing cells
            if config.track_n_vs_t:
                n_vs_t.append( [t, n[0], n[1]] )

                u_vs_t.append( [t, u[0], u[1]] )

            # implement cell division in saved lineages
            if tracking_lineage:
                lineages.divide_cell(mother_cell_id,daughter_cell_id_list,daughter_is_dividing_list,t)

            # check if lineage needs to start being tracked
            if track_lineage:
                if (t > config.track_lineage_time_interval[0]) and (t < config.track_lineage_time_interval[1]) and (not tracking_lineage):
                    tracking_lineage=True
                    for cell in cell_list:
                        lineages.add_lineage(cell.id, config.track_lineage_time_interval[0], cell.comp, True)

            # check if lineage tracking should stop
            if tracking_lineage:
                if t >= config.track_lineage_time_interval[1]:
                    tracking_lineage=False

        else:
            # next division would be after t_sim, stop simulation
            cont=False

            run_ended_early=False
            n_exploded=False
            t_end = config.t_sim

            # final save of number of dividing cells
            if config.track_n_vs_t:
                n_vs_t.append([t_end, n[0], n[1]])
                u_vs_t.append([t_end, u[0], u[1]])

        if len(cell_list)==0:
            # if no dividing cells left, stop simulation
            cont=False
            run_ended_early=True
            n_exploded=False
            t_end=t
            # adjust moments
            moment_data.adjust_moment_data(config.t_sim-t, n)

        if n[0] + n[1] >= config.n_max:
            # if more than x dividing cells, stop simulation
            cont=False
            run_ended_early=False
            n_exploded=True
            t_end=t
            # adjust moments
            moment_data.adjust_moment_data(config.t_sim - t, n)

        # increase event counter
        n_events+=1

    # save data
    run_stats = RunStats(run_ended_early=run_ended_early, t_end=t_end, n_exploded=n_exploded)
    output = SimulationResults(moments=moment_data, run_stats=run_stats)
    if config.track_n_vs_t:
        output.n_vs_t = np.array(n_vs_t)
        output.u_vs_t = np.array(u_vs_t)
    if track_lineage:
        output.lineages = lineages
    # and return as output
    return output
