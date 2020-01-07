import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
import matplotlib as mpl
import random
mpl.rcParams['pdf.fonttype'] = 42

from two_compartment_model_lib import run_sim
import tools

def load_data():
    sim_data = []
    for i in range(0,72):
        filename = f"two_comp_sweep_data/sweep_fixed_S30_Np40_i{i}.p"
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
plot_param = [ [0.2,-0.9,1.0], [0.2,-0.2,1.0], [0.9,-0.9,1.0], [0.9, -0.2, 1.0],
                [0.5,-0.5,0.6], [0.2,-0.2,0.3] ]

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

fig = plt.figure(figsize=(12, 8))

cmap = plt.cm.jet
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist[0] = (.7, .7, .7, 1.0) # force the first color entry to be grey
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

plt.subplot2grid((30,3),(4,0), rowspan=11)
min_val = 0
max_val = 1
phi = 1
N,M,D,C,C_t,MEAN,N_CV,M_CV,D_CV = tools.plot_alphas_for_constant_phi(phi,sim_data,alpha_n_range,alpha_m_range,phi_range,Np)
half_step = (alpha_m_range[1]-alpha_m_range[0])/2
im = plt.imshow(D_CV,interpolation='nearest',extent=(alpha_m_range[0]-half_step,alpha_m_range[-1]+half_step, 
                                                     alpha_n_range[-1]+half_step,alpha_n_range[0]-half_step), 
           cmap=cmap,vmin=min_val, vmax=max_val)
plt.title('phi=%.01f' % phi)
plt.ylabel('alpha_n')
plt.xlabel('alpha_m')
plt.xlim([alpha_m_range[0]-half_step,alpha_m_range[-1]+half_step])
plt.ylim([alpha_n_range[-1]+half_step,alpha_n_range[0]-half_step])
plt.xticks([-0.2,-0.4,-0.6,-0.8,-1])
#plt.yticks([0.025,0.05])

for n in range(0,4):
    i = plot_run_ind_list[n]
    plt.plot( sim_data[i][0]['alpha'][1], sim_data[i][0]['alpha'][0], '.r')
    plt.text( sim_data[i][0]['alpha'][1]+0.02, sim_data[i][0]['alpha'][0]+0.08, '%d' % (n+1), color='r' )
plt.plot([-1.05,0],[1.05,0],':r')    


plt.subplot2grid((30,3),(0,0), rowspan=1)
ax = plt.gca()
ax.set_position([0.163, 0.87, 0.153, 0.02])
cbar = fig.colorbar(im,cax=ax,orientation="horizontal")
cbar.set_ticks([0,.2,.4,.6,.8,1])
plt.title('Coefficient of variation')

plt.subplot2grid((30,3),(19,0), rowspan=11)
N,M,D,C,C_t,MEAN,N_CV,M_CV,D_CV = tools.plot_opposite_alphas(sim_data,alpha_n_range,alpha_m_range,phi_range,Np)
plt.imshow(D_CV,interpolation='nearest',extent=(alpha_n_range[0]-half_step,alpha_n_range[-1]+half_step,
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

for n in [1,2,4,5]:
    i = plot_run_ind_list[n]
    plt.plot( sim_data[i][0]['alpha'][0], sim_data[i][0]['phi'][0], '.r')
    plt.text( sim_data[i][0]['alpha'][0]-0.02, sim_data[i][0]['phi'][0]-0.08, '%d' % (n+1), color='r' )


plt.show()

# plot trajectories
    
np.random.seed(15)

t_lineage_range=[100,150]
#t_lineage_range=[200,250]

for n in range(0,len(plot_run_ind_list)):
    i=plot_run_ind_list[n]

    # set model parameters
    S=int( np.round(sim_data[i][0]['S']) )
    
    alpha=sim_data[i][0]['alpha']
    phi=sim_data[i][0]['phi'][0]
    N_0=int( alpha[0]*S )
    D = S*np.log(1+alpha[0])*(alpha[1]-alpha[0])/alpha[1]
    M_0 = int(np.round(D-N_0))
    
    params = {'S':S, 'alpha':alpha, 'phi':[phi,phi], 'T':[16.375159506031768,3.2357834505600382] }
    
    print("%d/%d, a_n:%1.1f, a_m:%1.1f, phi:%1.1f, S:%1.1f, N:%1.1f" % (i,len(sweep_param),alpha[0],alpha[1],phi,S,N_0) )
    
    # run simulations
    t_sim=1e3
    res = run_sim( t_sim,100000, params, n0=[N_0,M_0], track_n_vs_t=True, track_lineage_time_interval=t_lineage_range )

    # plot cellnumber dynamics vs time
    plt.subplot2grid((6,3),(n,1), rowspan=1)
    plt.plot([0,t_sim],[D,D],'--', color='grey')
    n_vs_t=res['n_vs_t']
    # plot total number of dividing cells in black
    plt.plot(n_vs_t[:,0],n_vs_t[:,1]+n_vs_t[:,2],'-k',lw=1)
    # plot number of dividing cells in stem cell compartment
    plt.plot(n_vs_t[:,0],n_vs_t[:,1],'-g',lw=1)
#    # plot number of dividing cells in transit amplifying compartment
    plt.plot(n_vs_t[:,0],n_vs_t[:,2],'-b',lw=1)
    plt.xlim([0,t_sim])
    plt.ylim([0,3*D])
    
    plt.yticks([0,round(D),round(2*D)],['0','D', '2D'])
    
    
    plt.text(10,2.5*D,'%d' % (n+1))
    
    if n!=5:
        plt.xticks([])
    else:
        plt.xlabel('time (hours)')
        
    if n==3:
        plt.ylabel('# of dividing cells')
                   
    # plot cell lineage
    
    L_list=res['Lineage']
    plt.subplot2grid((6,3),(n,2), rowspan=1)
    x0=0
    max_lin = min(10,len(L_list))
    random.seed(1)
    ra_L = random.choices(L_list, k = max_lin)
    for i in ra_L:
        w = i.draw_lineage(t_lineage_range[1],x0,show_cell_id=False)
        x0+=w+2
    plt.ylim([t_lineage_range[1],t_lineage_range[0]])
    plt.xlim([-2,x0])
    plt.xticks([])
    plt.yticks([])

    if n==3:
        plt.ylabel('time')

plt.show()
