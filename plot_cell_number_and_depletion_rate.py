import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'  # export text as text in SVG, not as paths

from stem_cell_model import tools


def load_data():
    sim_data = []
    for i in range(0,221):
        filename = f"two_comp_sweep_data_fixed_D_aT1/sweep_i{i}.p"
        sim_data.extend(pickle.load( open( filename, "rb" ) ))
    return sim_data
sim_data = load_data()
Np=40
start = .025
end = 1

alpha_n_range = np.linspace(start,end,Np)
alpha_m_range = np.linspace(-end,-start,Np)
phi_range = np.linspace(start,end,Np)

cmap = plt.cm.jet
cmaplist_len = 512  # Must be higher than the actual max value, otherwise the value 1 will be gray too, instead of only 0
cmaplist = [cmap(i / cmaplist_len) for i in range(cmaplist_len)]
cmaplist[0] = (.7, .7, .7, 1.0) # force the first color entry to be grey
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmaplist_len)

#%%
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

#%% 

fig = plt.figure(figsize=(8, 8))
half_step = (alpha_m_range[1]-alpha_m_range[0])/2

plt.subplot2grid((30,2),(4,0), rowspan=11)
min_val = 0
max_val = 400
phi = 1
result_images = tools.plot_alphas_for_constant_phi(phi, sim_data, alpha_n_range, alpha_m_range, phi_range, Np)
im = plt.imshow(result_images.S,interpolation='nearest',extent=(alpha_m_range[0]-half_step,alpha_m_range[-1]+half_step,
                                                     alpha_n_range[-1]+half_step,alpha_n_range[0]-half_step), 
                                                     cmap=cmap,vmin=min_val, vmax=max_val)
plt.title('phi=%.01f' % phi)
plt.ylabel('alpha_n')
plt.xlabel('alpha_m')
plt.xlim([alpha_m_range[0]-half_step,alpha_m_range[-1]+half_step])
plt.ylim([alpha_n_range[-1]+half_step,alpha_n_range[0]-half_step])
plt.xticks([-0.2,-0.4,-0.6,-0.8,-1])

for n in range(0,4):
    i = plot_run_ind_list[n]
    plt.plot( sim_data[i][0]['alpha'][1], sim_data[i][0]['alpha'][0], '.r')
    plt.text( sim_data[i][0]['alpha'][1]+0.02, sim_data[i][0]['alpha'][0]+0.08, '%d' % (n+1), color='r' )
plt.plot([-1.05,0],[1.05,0],':r')    


plt.subplot2grid((30,2),(0,0), rowspan=1)
ax = plt.gca()
ax.set_position([0.163, 0.87, 0.278, 0.02])
cbar = fig.colorbar(im,cax=ax,orientation="horizontal")
cbar.set_ticks([0,100,200,300,400])
plt.title('Number of cells in proliferation compartment')

plt.subplot2grid((30,2),(19,0), rowspan=11)
result_images = tools.plot_opposite_alphas(sim_data, alpha_n_range, alpha_m_range, phi_range, Np)
plt.imshow(result_images.S,interpolation='nearest',extent=(alpha_n_range[0]-half_step,alpha_n_range[-1]+half_step,
                                                phi_range[-1]+half_step,phi_range[0]-half_step), 
                                                cmap=cmap,vmin=min_val, vmax=max_val)
plt.xlabel('alpha_n, -alpha_m')
plt.ylabel('phi')
plt.ylim([phi_range[0]-half_step,phi_range[-1]+half_step])
plt.xlim([alpha_n_range[-1]+half_step,alpha_n_range[0]-half_step])
plt.title('sigma_D for alpha_n=-alpha_m')
plt.xticks([0.2,0.4,0.6,0.8,1])


for n in [1,2,4,5]:
    i = plot_run_ind_list[n]
    plt.plot( sim_data[i][0]['alpha'][0], sim_data[i][0]['phi'][0], '.r')
    plt.text( sim_data[i][0]['alpha'][0]-0.02, sim_data[i][0]['phi'][0]-0.08, '%d' % (n+1), color='r' )

#%%

plt.subplot2grid((30,2),(4,1), rowspan=11)
min_val = -.1
max_val = 2
phi = 1
result_images = tools.plot_alphas_for_constant_phi(phi, sim_data, alpha_n_range, alpha_m_range, phi_range, Np)
im = plt.imshow(result_images.C,interpolation='nearest',extent=(alpha_m_range[0]-half_step,alpha_m_range[-1]+half_step,
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


plt.subplot2grid((30,2),(0,1), rowspan=1)
ax = plt.gca()
cbar = fig.colorbar(im,cax=ax,orientation="horizontal")
ax.set_position([0.585, 0.87, 0.278, 0.02])
cbar.set_ticks([0,1,2])
plt.title('Depletion rate (events/1,000h)')

plt.subplot2grid((30,2),(19,1), rowspan=11)
result_images = tools.plot_opposite_alphas(sim_data, alpha_n_range, alpha_m_range, phi_range, Np)
plt.imshow(result_images.C,interpolation='nearest',extent=(alpha_n_range[0]-half_step,alpha_n_range[-1]+half_step,
                                                phi_range[-1]+half_step,phi_range[0]-half_step), 
                                                cmap=cmap,vmin=min_val, vmax=max_val)
plt.xlabel('alpha_n, -alpha_m')
plt.ylabel('phi')
plt.ylim([phi_range[0]-half_step,phi_range[-1]+half_step])
plt.xlim([alpha_n_range[-1]+half_step,alpha_n_range[0]-half_step])
plt.title('sigma_D for alpha_n=-alpha_m')
plt.xticks([0.2,0.4,0.6,0.8,1])

for n in [1,2,4,5]:
    i = plot_run_ind_list[n]
    plt.plot( sim_data[i][0]['alpha'][0], sim_data[i][0]['phi'][0], '.r')
    plt.text( sim_data[i][0]['alpha'][0]-0.02, sim_data[i][0]['phi'][0]-0.08, '%d' % (n+1), color='r' )

plt.show()
