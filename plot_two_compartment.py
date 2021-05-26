import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
import matplotlib as mpl
import random
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

from stem_cell_model.two_compartment_model_space import run_sim_niche
from stem_cell_model import tools


def load_data():
    sim_data = []
    for i in range(0,222):
        filename = f"two_comp_sweep_data_fixed_D_aT100/sweep_i{i}.p"
        sim_data.extend(pickle.load( open( filename, "rb" ) ))
    return sim_data
sim_data = load_data()
Np=40
start = .025
end = 1

alpha_n_range = np.linspace(start,end,Np)
alpha_m_range = np.linspace(-end,-start,Np)
phi_range = np.linspace(start,end,Np)

# parameters (alpha_n, alpha_m and phi) for which trajectories will be plotted
plot_param = [[0.95, -0.95, 1.0], [0.2, -0.95, 1.0], [0.2, -0.2, 1.0], [0.2, -0.2, 0.25]]

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

fig = plt.figure(figsize=(7, 6))

cmap = plt.cm.jet
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist[0] = (.7, .7, .7, 1.0) # force the first color entry to be grey
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

plt.subplot2grid((30,3),(4,0), rowspan=11)
min_val = 0
max_val = 1
phi = 1
result_images = tools.plot_alphas_for_constant_phi(phi, sim_data, alpha_n_range, alpha_m_range, phi_range, Np)
half_step = (alpha_m_range[1]-alpha_m_range[0])/2
im = plt.imshow(result_images.D_CV,interpolation='nearest',extent=(alpha_m_range[0]-half_step,alpha_m_range[-1]+half_step,
                                                     alpha_n_range[-1]+half_step,alpha_n_range[0]-half_step), 
           cmap=cmap,vmin=min_val, vmax=max_val)
plt.title('phi=%.01f' % phi)
plt.ylabel('alpha_n')
plt.xlabel('alpha_m')
plt.xlim([alpha_m_range[0]-half_step,alpha_m_range[-1]+half_step])
plt.ylim([alpha_n_range[-1]+half_step,alpha_n_range[0]-half_step])
plt.xticks([-0.2,-0.4,-0.6,-0.8,-1])
#plt.yticks([0.025,0.05])

for n in [1,2,0]:
    i = plot_run_ind_list[n]
    plt.plot( sim_data[i][0]['alpha'][1], sim_data[i][0]['alpha'][0], '.r')
    plt.text( sim_data[i][0]['alpha'][1]+0.02, sim_data[i][0]['alpha'][0]+0.08, '%d' % (n+1), color='r' )
plt.plot([-1.05,0],[1.05,0],':r')    


plt.subplot2grid((30,3),(0,0), rowspan=1)
ax = plt.gca()
cbar = fig.colorbar(im,cax=ax,orientation="horizontal")
cbar.set_ticks([0,.2,.4,.6,.8,1])
plt.title('Coefficient of variation')

plt.subplot2grid((30,3),(19,0), rowspan=11)
result_images = tools.plot_opposite_alphas(sim_data, alpha_n_range, alpha_m_range, phi_range, Np)
plt.imshow(result_images.D_CV,interpolation='nearest',extent=(alpha_n_range[0]-half_step,alpha_n_range[-1]+half_step,
                                                phi_range[-1]+half_step,phi_range[0]-half_step), 
           cmap=cmap,vmin=min_val, vmax=max_val)
plt.xlabel('alpha_n, -alpha_m')
plt.ylabel('phi')
plt.ylim([phi_range[0]-half_step,phi_range[-1]+half_step])
plt.xlim([alpha_n_range[-1]+half_step,alpha_n_range[0]-half_step])
#plt.xlim([1.05,0.1-0.05])
#plt.ylim([0.1-0.05,1.05])
plt.title('sigma_D for alpha_n=-alpha_m')
plt.xticks([0.2,0.4,0.6,0.8,1])

for n in [2,0,3]:
    i = plot_run_ind_list[n]
    plt.plot( sim_data[i][0]['alpha'][0], sim_data[i][0]['phi'][0], '.r')
    plt.text( sim_data[i][0]['alpha'][0]-0.02, sim_data[i][0]['phi'][0]-0.08, '%d' % (n+1), color='r' )


# plot trajectories

D = 30
t_lineage_range=[100,160]
seeds = [1, 6, 1, 3]

for n in range(0,len(plot_run_ind_list)):
    i=plot_run_ind_list[n]

    # set model parameters
    S=int( np.round(sim_data[i][0]['S']) )
    
    alpha=sim_data[i][0]['alpha']
    a=sim_data[i][0]['a']
    phi=sim_data[i][0]['phi'][0]
    N_0=int( alpha[0]*S )
    M_0 = int(np.round(D-N_0))
    
    params = {'S':S, 'alpha':alpha, 'phi':[phi,phi], 'T':[16.375159506031768,3.2357834505600382], 'a':a}
    np.random.seed(seeds[n])
    
    print("%d/%d, a_n:%1.1f, a_m:%1.1f, phi:%1.1f, S:%1.1f, N:%1.1f" % (i,len(sweep_param),alpha[0],alpha[1],phi,S,N_0) )
    
    # run simulations
    t_sim=1e3
    res = run_sim_niche( t_sim,100000, params, n0=[N_0,M_0], track_n_vs_t=True, track_lineage_time_interval=t_lineage_range )

    # plot cellnumber dynamics vs time
    plt.subplot2grid((4,5),(n,2), rowspan=1, colspan=3)
    plt.plot([0,t_sim],[D,D],'--', color='grey')
    n_vs_t=res['n_vs_t']
    # plot total number of dividing cells in black
    plt.plot(n_vs_t[:,0],n_vs_t[:,1]+n_vs_t[:,2],'-k',lw=1)
    # plot number of dividing cells in stem cell compartment
    plt.plot(n_vs_t[:,0],n_vs_t[:,1],color="#7FB840", lw=1)
#    # plot number of dividing cells in transit amplifying compartment
    plt.plot(n_vs_t[:,0],n_vs_t[:,2],color="#009CB5",lw=1)
    plt.xlim([0,t_sim])
    plt.ylim([0,2*D])
    
    plt.yticks([0, D, 2*D])
    
    
    plt.text(10,1.6*D,'%d' % (n+1))
    
    if n!=3:
        plt.xticks([])
    else:
        plt.xlabel('time (hours)')
        
    if n==1:
        plt.ylabel('# of dividing cells')

plt.show()
