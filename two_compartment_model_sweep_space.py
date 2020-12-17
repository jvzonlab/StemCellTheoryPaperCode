import numpy as np
import pickle as pickle

from two_compartment_model_lib import run_sim_niche

def main():
    #%% set sweep parameters
    
    # fix total number of dividing cells (measured values: anything from 10 to 50 dividing cells in a crypt)
    D=30

    # Fix swapping parameter
    a=0.1
    
    # fix cell cycle parameters (based on measured values)
    T=[16.153070175438597,3.2357834505600382]
    
    # number of steps in sweep
    Np=40
    start = 25
    end = 1000
    
    # total simulation time
    t_sim=1e5
    
    # sweep alpha for stem cell compartment from 0.1 to 1.0
    alpha_n=np.linspace(start,end,Np)*.001
    
    # sweep alpha for transit amplifying compartment from -1.0 to -0.1
    alpha_m=np.linspace(-end,-start,Np)*.001
    
    # sweep degree of symmetry phi from 0.1 to 1.0
    phi_r=np.linspace(start,end,Np)*.001   
    
    n_max = 10000000 #if counting overgrowth of dividing cells
    
    sweep_params, sweep_n0 =  get_param_list(alpha_n,alpha_m,phi_r,T,D,a)    
    
    sweep_params_split, sweep_n0_split = break_list(sweep_params, sweep_n0, 72)
    for it in range(len(sweep_params_split)):
        print(f'iteration {it}')
        perform_sweep(sweep_params_split[it],sweep_n0_split[it],
                        t_sim,n_max,f'two_comp_sweep_data_fixed_D_a{a}/sweep_fixed_D30_Np40_a{a}_i{it}.p')

    # perform_sweep(sweep_params,sweep_n0,t_sim,n_max,'two_comp_sweep_data.p')
   
def break_list(sweep_params, sweep_n0, c):
    sweep_params_split = split_list(sweep_params,c)
    sweep_n0_split = split_list(sweep_n0,c)
    return sweep_params_split, sweep_n0_split
def split_list(alist,parts):
    a = [[alist[j] for j in list(filter(lambda x: x%parts == i, range(0,len(alist))))] for i in range(0,parts)]
    return a
#%% calculate simulation parameters for each simulation in sweep

def get_param_list(alpha_n,alpha_m,phi_r,T,D,a):
    sweep_params=[]
    sweep_n0=[]
    
    # for each combination of alpha_n, alpha_m and phi
    for a_n in alpha_n:
        for a_m in alpha_m:
            for phi in phi_r:
                # calculate division probabilities p_i and q_i in compartment i=n,m
                p_n = (phi + a_n)/2
                q_n = (phi - a_n)/2
                p_m = (phi + a_m)/2
                q_m = (phi - a_m)/2
                
                # check if division probabilities exist for this alpha_n, alpha_m and phi combination
                if (p_n>=0) and (q_n>=0) and (p_m>=0) and (q_m>=0):
                    # calculate compartment size S
                    S = D/np.log(1+a_n)*a_m/(a_m-a_n)
                    # calculate the average number of dividing cells in stem cell compartment
                    N_avg = a_n*S
                    # calculate the average number of dividing cells in transit amplifying compartment
                    M_avg = D - N_avg
                    
                    # save parameters
                    sweep_params.append( {'S':int(np.round(S)), 'alpha':[a_n, a_m], 'phi':[phi,phi], 'T':T, 'a':a } )
                    # and initial conditions
                    sweep_n0.append( [ int(np.round(N_avg)), int(np.round(D-N_avg)) ] )
    return sweep_params, sweep_n0
#%% perform sweep

def perform_sweep(sweep_params,sweep_n0,t_sim,n_max,filename):                
    np.random.seed(1)   
    
    sim_data=[]
    for i in range(0,len(sweep_params)):
        
        # print run information
        print("%d/%d, a_n:%1.3f, a_m:%1.3f, phi:%1.3f, S:%d, N0:%d, M0:%d" % 
              (i,len(sweep_params), sweep_params[i]['alpha'][0],sweep_params[i]['alpha'][1],
               sweep_params[i]['phi'][0],sweep_params[i]['S'],
               sweep_n0[i][0],sweep_n0[i][1]) )
    
        # some simulation will end before the total simulation time t_sim because
        # stem cells are fully lost. In that case, we rerun simulations with the same 
        # initial conditions but a different random seed until the total simulation time
        # has exceeded t_sim
        
        # initialize simulation counter
        c=0
    
        # intialize statistic quantities for entire run
        nm_mean=np.zeros(2)
        nm_sq=np.zeros(2)
        nm_prod=0
    
        # analyze the total simulated time to zero    
        t_tot=0
        # set # of runs that ended early to zero    
        n_runs_ended_early=0
        
        # start simulation
        cont=True
        while cont:
            # do simulation for a time (t_sim - t_tot)
            res = run_sim_niche( t_sim-t_tot,n_max, sweep_params[i], n0=sweep_n0[i], track_n_vs_t=False)
            
            # add simulated time to total time
            t_tot += res['RunStats']['t_end']
            # and accumulate statistics for each individual run
            nm_mean += res['Moments']['mean']
            nm_sq += res['Moments']['sq']
            nm_prod += res['Moments']['prod']
        
#            print ("simulated run for t=%f, total accumulated time: %f" % (res['RunStats']['t_end'], t_tot) )
            
            # if run ended before time (t_sim - t_tot)
            if res['RunStats']['run_ended_early']==True:
                # store this
                n_runs_ended_early += 1
                # and continue
                cont=True
            else:
                # stop simulating for these parameter values
                cont=False
        
        # save simulation data
        output={'mean':nm_mean,'sq':nm_sq,'prod':nm_prod,'t_tot':t_tot,'n_runs_ended_early':n_runs_ended_early}    
        sim_data.append( [sweep_params[i], output] )
    
        # print run statistics
        nm_mean = nm_mean/t_tot
        nm_std = nm_sq/t_tot - nm_mean**2
        cc_NM = nm_prod/t_tot - nm_mean[0]*nm_mean[1]
        
        print("\t<N>=%f, s_N=%f" % (nm_mean[0], np.sqrt(nm_std[0]) ) )
        print("\t<M>=%f, s_M=%f" % (nm_mean[1], np.sqrt(nm_std[1]) ) )
        print("\t<N M>=%f" % cc_NM )
        print("\t<D>=%f, s_D=%f" % (nm_mean[0]+nm_mean[1], np.sqrt( nm_std[0]+nm_std[1]+2*cc_NM)) )

    # save all data in pickle file
    pickle.dump(sim_data, open( filename, "wb" ))

main()