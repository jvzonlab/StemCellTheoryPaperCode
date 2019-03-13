import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle

from two_compartment_model_lib import run_sim
from lineage_lib import Lineage

#%% initialize plot

plot_data=[]

# generate parameter values for sweep
D=30
T=[15.,2.3]
Np=10
alpha_n_range=np.linspace(0.1,1,Np)
alpha_m_range=np.linspace(-1,-0.1,Np)
phi_range=np.linspace(0.1,1,Np)

# load sweep data
sim_data = pickle.load( open( "sweep_data.p", "rb" ) )

# parameters (alpha_n, alpha_m and phi) for which trajectories will be plotted
plot_param = [ [0.2,-0.9,1.0], [0.2,-0.2,1.0], [0.9,-0.9,1.0], [0.9, -0.2, 1.0],
                [0.5,-0.5,0.6], [0.2,-0.2,0.2] ]

plot_run_ind_list=[]
# for each parameter set in plot_param
for s in range(0,len(plot_param)):
    # determine index for run with simulation parameters closest to plot_param[s]
    d_min=6e66
    ind_min=-1
    for i in range(0,len(sim_data)):
        # get parameters for this run
        sweep_param=sim_data[i][0]
        # look at difference in alpha_n,m and phi
        da0 = sweep_param['alpha'][0]-plot_param[s][0]
        da1 = sweep_param['alpha'][1]-plot_param[s][1]
        dph = sweep_param['phi'][0]-plot_param[s][2]
        # if smaller than current minimum distance
        if (da0**2+da1**2+dph**2)<d_min:
            # set new minimum to this distance
            d_min=da0**2+da1**2+dph**2
            # and save index of run
            ind_min=i
    # save index of overal minimum            
    plot_run_ind_list.append( ind_min )

#%% calculate statistical results for fixed phi

# value of phi to plot
phi=1.0

# initialize arrays to store results
N_std=np.zeros((Np,Np))
M_std=np.zeros((Np,Np))
D_std=np.zeros((Np,Np))
f_collapse=np.zeros((Np,Np))

# for all runs
for s in sim_data:
    sweep_param = s[0]
    run_data = s[1]
    # if they were run at the correct phi
    if sweep_param['phi'][0]==phi:
        # calculate statistics for current run
        n_mean = run_data['mean']/run_data['t_tot']
        n_std_sq = run_data['sq']/run_data['t_tot'] - n_mean**2
        cc_NM = run_data['prod']/run_data['t_tot'] - n_mean[0]*n_mean[1]
        D_std_sq = n_std_sq[0]+n_std_sq[1]+2*cc_NM
        
        # get alpha
        alpha=sweep_param['alpha']
        # find indeces i and j corresponding to the current parameters alpha_n,m        
        i = np.where(np.abs(alpha[0]-alpha_n_range)==np.abs((alpha[0]-alpha_n_range)).min())[0][0]
        j = np.where(np.abs(alpha[1]-alpha_m_range)==np.abs((alpha[1]-alpha_m_range)).min())[0][0]
        
        # and store the results
        N_std[i,j]=np.sqrt( n_std_sq[0] )
        M_std[i,j]=np.sqrt( n_std_sq[1] )
        D_std[i,j]=np.sqrt( D_std_sq )
        f_collapse[i,j]=run_data['n_runs_ended_early']/run_data['t_tot']
        
#        print("<N>=%f, s_N=%f" % (n_mean[0], np.sqrt(n_std[0]) ) )
#        print("<M>=%f, s_M=%f" % (n_mean[1], np.sqrt(n_std[1]) ) )
#        print("<N M>=%f" % cc_NM )
#        print("<D>=%f, s_D=%f" % (n_mean[0]+n_mean[1], np.sqrt( n_std[0]+n_std[1]+2*cc_NM)) )

# add data here
plot_data.append( [N_std, M_std, D_std, f_collapse])

#%% calculate statistical quantities for alpha_n = - alpha_m

# initialize arrays to store results
N_std=np.zeros((Np,Np))
M_std=np.zeros((Np,Np))
D_std=np.zeros((Np,Np))
f_collapse=np.zeros((Np,Np))

# for all runs
for s in sim_data:
    sweep_param = s[0]
    run_data = s[1]
    # get alpha_n,m and phi
    alpha=sweep_param['alpha']
    phi=sweep_param['phi'][0]
    # if alpha_n = -alpha_m
    if np.abs(alpha[0]+alpha[1])<1e-3:
        # calculate statistics for current run
        n_mean = run_data['mean']/run_data['t_tot']
        n_std_sq = run_data['sq']/run_data['t_tot'] - n_mean**2
        cc_NM = run_data['prod']/run_data['t_tot'] - n_mean[0]*n_mean[1]

        # get indeces i and j for current parameters alpha_n and phi
        i = np.where(np.abs(phi-phi_range)==np.abs((phi-phi_range)).min())[0][0]
        j = np.where(np.abs(alpha[0]-alpha_n_range)==np.abs((alpha[0]-alpha_n_range)).min())[0][0]
        
        # and store the results
        N_std[i,j]=np.sqrt( n_std_sq[0] )
        M_std[i,j]=np.sqrt( n_std_sq[1] )
        D_std[i,j] = np.sqrt( n_std_sq[0]+n_std_sq[1]+2*cc_NM )
        f_collapse[i,j]=run_data['n_runs_ended_early']/run_data['t_tot']

# add data here
plot_data.append( [N_std, M_std, D_std, f_collapse])

#%% plot variability in # of dividing cells

plt.figure(1)

plt.subplot(2,3,1)

D_std=plot_data[0][2]
plt.imshow(D_std,interpolation='nearest',extent=(alpha_m_range[0]-0.05,alpha_m_range[-1]+0.05,alpha_n_range[-1]+0.05,alpha_n_range[0]-0.05), cmap='gray_r')
plt.title('sigma_D for phi=1')
plt.ylabel('alpha_n')
plt.xlabel('alpha_m')
plt.xlim([-1.05,-0.1+0.05])
plt.ylim([1.05,0.1-0.05])

for n in range(0,4):
    i = plot_run_ind_list[n]
    plt.plot( sim_data[i][0]['alpha'][1], sim_data[i][0]['alpha'][0], '.r')
    plt.text( sim_data[i][0]['alpha'][1]+0.02, sim_data[i][0]['alpha'][0]+0.08, '%d' % n, color='r' )

plt.plot([-1.05,0],[1.05,0],':r')

plt.subplot(2,3,4)

D_std=plot_data[1][2]
plt.imshow(D_std,interpolation='nearest',extent=(alpha_n_range[0]-0.05,alpha_n_range[-1]+0.05,phi_range[-1]+0.05,phi_range[0]-0.05), cmap='gray_r')
plt.title('sigma_D for alpha_n=-alpha_m')
plt.xlabel('alpha_n, -alpha_m')
plt.ylabel('phi')
plt.xlim([1.05,0.1-0.05])
plt.ylim([0.1-0.05,1.05])

for n in [1,2,4,5]:
    i = plot_run_ind_list[n]
    plt.plot( sim_data[i][0]['alpha'][0], sim_data[i][0]['phi'][0], '.r')
    plt.text( sim_data[i][0]['alpha'][0]-0.02, sim_data[i][0]['phi'][0]-0.08, '%d' % n, color='r' )

#%% plot trajectories
    
np.random.seed(4)

t_lineage_range=[100,150]

for n in range(0,len(plot_run_ind_list)):
    i=plot_run_ind_list[n]

    # set model parameters
    S=int( np.round(sim_data[i][0]['S']) )
    alpha=sim_data[i][0]['alpha']
    phi=sim_data[i][0]['phi'][0]
    N_0=int( alpha[0]*S )
    params = {'S':S, 'alpha':alpha, 'phi':[phi,phi], 'T':[15.,2.3] }
    
    print("%d/%d, a_n:%1.1f, a_m:%1.1f, phi:%1.1f, S:%1.1f, N:%1.1f" % (i,len(sweep_param),alpha[0],alpha[1],phi,S,N_0) )
    
    # run simulations
    t_sim=1e3
    res = run_sim( t_sim, params, n0=[N_0,D-N_0], track_n_vs_t=True, track_lineage_time_interval=t_lineage_range )

    # plot cellnumber dynamics vs time
    plt.subplot(6,3,2+3*n )
    n_vs_t=res['n_vs_t']
    # plot total number of dividing cells in black
    plt.plot(n_vs_t[:,0],n_vs_t[:,1]+n_vs_t[:,2],'-k',lw=1)
    # plot number of dividing cells in stem cell compartment
    plt.plot(n_vs_t[:,0],n_vs_t[:,1],'-b',lw=1)
    # plot number of dividing cells in transit amplifying compartment
    plt.plot(n_vs_t[:,0],n_vs_t[:,2],'-r',lw=1)
    plt.xlim([0,1e3])
    plt.ylim([0,70])
    plt.text(10,50,'%d' % n)
    
    if n!=5:
        plt.xticks([])
    else:
        plt.xlabel('time (hours)')
        
    if n==3:
        plt.ylabel('# of dividing cells')
            
    # plot cell lineage
    
    L_list=res['Lineage']
    plt.subplot(6,3,3+3*n )
    x0=0
    for i in range(0,10):
        w = L_list[i].draw_lineage(t_lineage_range[1],x0,show_cell_id=False)
        x0+=w+2
    plt.ylim([t_lineage_range[1],t_lineage_range[0]])
    plt.xlim([-2,x0])
    plt.xticks([])
    plt.yticks([])

    if n==3:
        plt.ylabel('time')

#%% plot remaining statistical quantities
        
plt.figure(2)        

ind_list=[0,1,3]
lbl_list=['N_std','M_std','f_collapse']

for m in range(0,3):
    
    plt.subplot(2,3,1+m)
    M=plot_data[0][ ind_list[m] ]
    plt.imshow(M,interpolation='nearest',extent=(alpha_m_range[0]-0.05,alpha_m_range[-1]+0.05,alpha_n_range[-1]+0.05,alpha_n_range[0]-0.05), cmap='gray_r')
    plt.title('%s for phi=1' % lbl_list[m] )
    plt.ylabel('alpha_n')
    plt.xlabel('alpha_m')
    plt.xlim([-1.05,-0.1+0.05])
    plt.ylim([1.05,0.1-0.05])

    for n in range(0,4):
        i = plot_run_ind_list[n]
        plt.plot( sim_data[i][0]['alpha'][1], sim_data[i][0]['alpha'][0], '.r')
        plt.text( sim_data[i][0]['alpha'][1]+0.02, sim_data[i][0]['alpha'][0]+0.08, '%d' % n, color='r' )
    
    plt.plot([-1.05,0],[1.05,0],':r')

    plt.subplot(2,3,4+m)
    M=plot_data[1][ ind_list[m] ]
    plt.imshow(M,interpolation='nearest',extent=(alpha_n_range[0]-0.05,alpha_n_range[-1]+0.05,phi_range[-1]+0.05,phi_range[0]-0.05), cmap='gray_r')
    plt.title('%s for alpha_n=-alpha_m' % lbl_list[m] )
    plt.xlabel('alpha_n, -alpha_m')
    plt.ylabel('phi')
    plt.xlim([1.05,0.1-0.05])
    plt.ylim([0.1-0.05,1.05])
    
    for n in [1,2,4,5]:
        i = plot_run_ind_list[n]
        plt.plot( sim_data[i][0]['alpha'][0], sim_data[i][0]['phi'][0], '.r')
        plt.text( sim_data[i][0]['alpha'][0]-0.02, sim_data[i][0]['phi'][0]-0.08, '%d' % n, color='r' )