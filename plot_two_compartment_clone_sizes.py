from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
import matplotlib as mpl

from stem_cell_model import sweeper, two_compartment_model_space
from stem_cell_model.lineages import CloneSizeDistribution
from stem_cell_model.parameters import SimulationParameters, SimulationConfig
from stem_cell_model.results import MultiRunStats

mpl.rcParams['pdf.fonttype'] = 42

def load_data() -> List[Tuple[SimulationParameters, MultiRunStats]]:
    return list(sweeper.load_sweep_results("two_comp_sweep_data_fixed_D_aT1"))

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
        da0 = sweep_param.alpha[0]-plot_param[s][0]
        da1 = sweep_param.alpha[1]-plot_param[s][1]
        dph = sweep_param.phi[0]-plot_param[s][2]
        # if smaller than current minimum distance
        if (da0**2+da1**2+dph**2)<d_min:
            # set new minimum to this distance
            d_min=da0**2+da1**2+dph**2
            # and save index of run
            ind_min=i
    # save index of overal minimum            
    plot_run_ind_list.append( ind_min )

fig = plt.figure(figsize=(4, 8))

cmap = plt.cm.jet
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist[0] = (.7, .7, .7, 1.0) # force the first color entry to be grey
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)


# plot trajectories
    
random = np.random.Generator(np.random.MT19937(seed=15))
t_lineage_range = (0, 160)
crypts = 50

for n in range(0,len(plot_run_ind_list)):
    i=plot_run_ind_list[n]

    # set model parameters
    params = sim_data[i][0]
    S = int(np.round(params.S))
    alpha = params.alpha
    a = params.a
    phi = params.phi[0]
    N_0 = params.n0[0]
    M_0 = params.n0[1]

    print("%d/%d, a_n:%1.1f, a_m:%1.1f, phi:%1.1f, S:%1.1f, N:%1.1f" % (i,len(sim_data),alpha[0],alpha[1],phi,S,N_0) )
    
    # run simulations
    t_sim=t_lineage_range[1] + 1
    clone_sizes = CloneSizeDistribution()
    clone_size_duration = 40
    min_clone_count_time, max_clone_count_time = t_lineage_range
    config = SimulationConfig(t_sim=t_sim, n_max=100000, track_n_vs_t=True, track_lineage_time_interval=t_lineage_range, random=random)

    for i in range(crypts):  # simulate a crypt 50 times, so that the uncertainty in the clone size is small

        res = two_compartment_model_space.run_simulation_niche(config, params)
        L_list=res.lineages
        for i in range(len(L_list)):
            clone_sizes.merge(L_list[i].get_clone_size_distributions_with_duration(min_clone_count_time, max_clone_count_time, clone_size_duration))

    # plot clone sizes
    max_clone_size = max(12, clone_sizes.max())  # Show at least 12 bins
    plt.subplot2grid((6,1),(n,0), rowspan=1)
    plt.text(1,1,"#" + str(n + 1), transform=plt.gca().transAxes, verticalalignment="top")
    plt.hist(clone_sizes.to_flat_array(), bins=np.arange(2, max_clone_size + 1) - 0.5, color="black")
    if n == 5:
        plt.xlabel(f'clone size for {clone_size_duration}h')
        plt.xticks(range(2, max_clone_size + 2, 2))
    else:
        plt.xticks([])
    if n == 3:
        plt.ylabel("count")

    # print sister and cousin statistics
    stats = None
    for i in range(len(L_list)):
        lineage = L_list[i]
        if stats is None:
            stats = lineage.count_divisions()
        else:
            stats += lineage.count_divisions()
    print(f"STATS FOR PARAMETER SET {n}: {stats}")

plt.show()
