import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
import time

from lineage_lib import Lineage

#from skewed distribution
ae = 6.104045529523038
loce = 12.242161056111865
scalee = 5.248997116207871

ages_list = []

# get cell age at the time of division
def get_division_age(T):
    global ages_list, ae, loce, scalee
    
    #generate a list of cell cycle durations by drawing numbers from skew normal distribution with parameters ae, loce, scalee from experimental data
    if len(ages_list) == 0:
        ages_list = skewnorm(ae, loce, scalee).rvs(1000000)
    #assign the last number in ages list as the age of cell and remove number from list
    a, ages_list = ages_list[-1], ages_list[:-1]

    return (a)

# cell class
class Cell:
    
    # initialize cell
    def __init__(self,cell_id,compartment,age,T):
        # unique cell identifier
        self.id = cell_id
        # set compartment that contains the cell
        self.comp = compartment
        # and current cell age
        self.age = age
        # and cell age when division will occur
        self.age_div = get_division_age(T)
        # make sure division does not occur before current time
        # (should be very rare for proper choise of T[0] and T[1])
        while self.age_div<self.age:
            self.age_div = get_division_age(T)
        
    # return time to next division
    def get_time_to_division(self):
        return (self.age_div-self.age)

# functions for calculating statistics of fluctuations of stem cell number
def init_moment_data():
    # <N>, <M>, <N^2>, <M^2>, <NM> 
    moment_data={'mean':np.zeros(2), 'sq':np.zeros(2), 'prod':0}
    return (moment_data)

def adjust_moment_data(dt,n, moment_data):
    # <N>,<M>
    moment_data['mean'] += dt*n
    # <N^2>,<M^2>
    moment_data['sq'] += dt*n**2
    # <NM>
    moment_data['prod'] += dt*n[0]*n[1]
    
    return (moment_data)

# main simulation function

# t_sim - total simulation time
# params - list containing cell cycle time <T>, stem cell compartment size <S>,
#       degrees of symmetry <phi_i> and growth rate <alpha_i> for compartment <i>
# n0_i - initial number of stem cells in compartment <i>
# track_lineage_time_interval - list [t_start, t_end] during which lineage information 
#        should be recorded. If empty, no lineage recorderd
# track_n_vs_t - track cell number versus time. If <false> only calculate moments
def run_sim( t_sim,n_max, params, n0=[0,0], track_lineage_time_interval=[], track_n_vs_t=False):
    
    # if an interval to track lineages is defined
    if len(track_lineage_time_interval)==2:
        # then set flag for tracking them
        track_lineage=True
    else:
        track_lineage=False

    ### init cells
    
    # initialize current cell id
    cell_id=1
    cell_list=[]
    # initialize dividing cells in stem cell compartment
    for n in range(0,n0[0]):
        age = params['T'][0]*np.random.rand()
        cell_list.append( Cell(cell_id,0,age,params['T']) )
        cell_id += 1
    # initialize dividing cells outside compartment
    for n in range(0,n0[1]):
        age = params['T'][0]*np.random.rand()
        cell_list.append( Cell(cell_id,1,age,params['T']) )
        cell_id += 1
        
    # get sorted list of dividing cells by increasing time of division
    cell_list.sort(key=lambda x: x.get_time_to_division() )

    ### calculate p,q parameters for both compartment
    
    p=[]
    q=[]
    for compartment in [0,1]:
        p.append( (params['phi'][compartment] + params['alpha'][compartment])/2 )
        q.append( (params['phi'][compartment] - params['alpha'][compartment])/2 )

    ### init for data collection
    
    # initalize simulation time
    t=0
    # intialize stem cell number n(t) = [ N(t), M(t) ], where N(t) is # stem cells 
    # in compartment 1 and M(t) is # of stem cells in compartment 2
    n = np.array( n0, dtype=int )
    # initialize differentiated cell number u_i(t) in compartment <i>
    u = np.array( [params['S']-n0[0], 0], dtype=int )
    
    # initialize moment data for fluctuations in stem cell number
    moment_data = init_moment_data()
    
    # run stats
    run_ended_early=False
    t_end=0
    n_events=0
    
    if track_n_vs_t:
        # initialize tracks of stem cell numbers N(t) and M(t)
        n_vs_t = [ [t, n[0], n[1]] ]
        # and same for numbers of differentiated cells
        u_vs_t = [ [t, u[0], u[1]] ]
    
    # if an interval to track lineages is defined
    if len(track_lineage_time_interval)==2:
        # then set flag for tracking them
        track_lineage=True
        # initialize lineage list
        L_list=[]
    else:
        track_lineage=False

    ### run simulation
    
    tracking_lineage=False
    cont=True
    while cont:
        # get time dt to next division
        dt = cell_list[0].get_time_to_division()
        
        # if time of division is before end of simulation
        if (t+dt<t_sim):
            # implement the next division
            
            # set new time
            t=t+dt
#            print("--- t=%f ---" % t)
            
            # adjust moments
            moment_data=adjust_moment_data(dt,n,moment_data)
            
            # add dt to age
            for cell in cell_list:
                cell.age += dt
        
            # get compartment of dividing cell
            compartment=cell_list[0].comp
            
            ### get type of division
            
            # draw random number in (0,1)
            r = np.random.rand()
            if r<=p[compartment]:
                # if r in (0,p), then div -> div + div
                div_type=0
            elif r<=(p[compartment]+q[compartment]):
                # if r in (p,p+q), then div -> non-div + non-div
                div_type=2
            else:
                # else, div -> div + non-div
                div_type=1
            
            if tracking_lineage:
                # if tracking lineage, get ids for mother and daughters
                daughter_cell_id_list=[]
                daughter_is_dividing_list=[]
                mother_cell_id = cell_list[0].id
    
            ### execute division
            if div_type==0:
                # div -> div + div
                # generate two new dividing cells to compartment <c>
                for i in [0,1]:
                    # add new cells to cell list
                    cell_list.append( Cell(cell_id,compartment,0,params['T']) )
                    if tracking_lineage:
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
                cell_list.append( Cell(cell_id,compartment,0,params['T']) )
                if tracking_lineage:
                    # rember id of this daughter, if tracking lineage
                    daughter_cell_id_list.append( cell_id )
                    daughter_is_dividing_list.append( True )
                cell_id += 1  
                if tracking_lineage:
                    # and remember id of the non-dividing daughter
                    daughter_cell_id_list.append( cell_id )
                    daughter_is_dividing_list.append( False )
                cell_id += 1
                # adjust number of non-dividing differentiated cells
                u[compartment] += 1
                
            elif div_type==2:
                # div -> non-div + non-div
                for i in [0,1]:
                    if tracking_lineage:
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
            del cell_list[0]
    
            # if division was in compartment 0
            if compartment==0:
                # get list of cells in current compartment    
                comp_cell_list = [x for x in cell_list if x.comp==0]
                # draw random cell in compartment
                n_move=int((params['S']+1)*np.random.rand())
                # if cell is a dividing cell
                if n_move<len(comp_cell_list):
                    random_cell = comp_cell_list[n_move]
                    # move it to the next compartment
                    random_cell.comp = 1
                    # implement cell moving in saved lineages
                    if tracking_lineage:
                        for L in L_list:
                            L.move_cell(random_cell.id, t, random_cell.comp)
                    # adjust number of dividing stem cells
                    n[0] -= 1
                    n[1] += 1
                    
                else:
                    # adjust number of non-dividing differentiated cells
                    u[0] -= 1
                    u[1] += 1
            
            # finally, get list of cells sorted by time to next division
            cell_list.sort(key=lambda x: x.get_time_to_division() )
    
            # save number of dividing cells
            if track_n_vs_t:
                n_vs_t.append( [t, n[0], n[1]] )
                
                u_vs_t.append( [t, u[0], u[1]] )
    
            # implement cell division in saved lineages
            if tracking_lineage:
                for L in L_list:
                    L.divide_cell(mother_cell_id,daughter_cell_id_list,daughter_is_dividing_list,t)
        
            # check if lineage needs to start being tracked
            if track_lineage:
                if (t>track_lineage_time_interval[0]) and (t<track_lineage_time_interval[1]) and (not tracking_lineage):
                    tracking_lineage=True
                    for cell in cell_list:
                        L_list.append( Lineage(cell.id, track_lineage_time_interval[0], cell.comp, True, 1) )
            
            # check if lineage tracking should stop
            if (tracking_lineage==True):
                if (t>=track_lineage_time_interval[1]):
#                    print("stop tracking lineage")
                    tracking_lineage=False
    
        else:
            # next division would be after t_sim, stop simulation
            cont=False
            # adjust moments
            moment_data=adjust_moment_data(t_sim-t,n,moment_data)
            run_ended_early=False
            n_exploded=False
            t_end=t_sim
            
        if len(cell_list)==0:
            # if no dividing cells left, stop simulation
            cont=False
            run_ended_early=True
            n_exploded=False
            t_end=t
#            # adjust moments
#            moment_data=adjust_moment_data(t_sim-t,n,moment_data)
        
        if n[0] + n[1] >= n_max:
            # if more than x dividing cells, stop simulation
            cont=False
            run_ended_early=False
            n_exploded=True
            t_end=t
#            # adjust moments
#            moment_data=adjust_moment_data(t_sim-t,n,moment_data)
    
        # increase event counter
        n_events+=1
    
    # save data
    output={'Moments':moment_data}
    if track_n_vs_t:
        output['n_vs_t']=np.array(n_vs_t)
        output['u_vs_t']=np.array(u_vs_t)
    if track_lineage:
        output['Lineage']=L_list
    output['RunStats']={'run_ended_early':run_ended_early,'t_end':t_end,'n_exploded':n_exploded}
    # and return as output
    return (output)
