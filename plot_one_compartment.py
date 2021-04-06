import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
from stem_cell_model import tools

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

# Collect simulation data from file
from stem_cell_model.two_compartment_model import run_sim

sim_data = pickle.load( open( "one_comp_sweep_data.p", "rb" ) )

alpha_range=np.linspace(-10,10,21)
alpha_range=np.array([x/10. for x in alpha_range])

phi_range=np.linspace(0,1,11)

# initialize arrays to store results
N=np.zeros((11))
M=np.zeros((11))
D=np.zeros((11))
C=np.zeros((11))
E=np.zeros((11))
C_t=np.zeros((11))
E_t=np.zeros((11))
MEAN=np.zeros((11))

div = np.zeros((11)) #coefficient of variation


for s in sim_data:
    sweep_param = s[0]
    run_data = s[1]

    alpha=sweep_param['alpha'][0]
    phi=sweep_param['phi'][0]

    single_run_statistics = tools.get_single_run_statistics(run_data)
    
    # get index i for current parameter phi
    i = np.where(phi_range == phi)[0][0]

    N[i] = single_run_statistics.n_std
    M[i] = single_run_statistics.m_std
    D[i] = single_run_statistics.d_std
    C[i] = single_run_statistics.f_collapse
    E[i] = single_run_statistics.n_explosions
    C_t[i] = single_run_statistics.f_collapse_t
    E_t[i] = single_run_statistics.n_explosions_t
    MEAN[i] = single_run_statistics.d_mean
    div[i] = single_run_statistics.d_coeff_var

# Plot some example counts of dividing cells over time, for different values of phi and alpha
S = 10
N_avg = 10
D = 30
T=[16.153070175438597,3.2357834505600382]  # Based on measured values

standard_params = {'S':int(np.round(S)), 'phi':[0.9, 0.9], 'T':T, 'alpha': [0, 0]}
np.random.seed(1)  # Fixed seed

# First for alpha = -0.5, phi = 1
t_alpha_example_sim = 50
simulations_alpha_negative = []
while len(simulations_alpha_negative) < 5:
    simulations_alpha_negative.append(run_sim(t_alpha_example_sim, 100000, {**standard_params, 'alpha': [-0.5, -0.5]}, n0=[N_avg, D - N_avg], track_n_vs_t=True))

# Then for alpha = 0.5, phi = 1
simulations_alpha_positive = []
while len(simulations_alpha_positive) < 5:
    simulations_alpha_positive.append(run_sim(t_alpha_example_sim, 100000, {**standard_params, 'alpha': [0.5, 0.5]}, n0=[N_avg, D - N_avg], track_n_vs_t=True))

# Then longer-term (so alpha = 0.1), first for phi = 0.9
t_phi_sim = 1000
simulations_phi_zero_point_nine = list()
while len(simulations_phi_zero_point_nine) < 5:
    simulations_phi_zero_point_nine.append(run_sim(t_phi_sim, 100000, {**standard_params, 'phi': [0.9, 0.9]}, n0=[N_avg, D - N_avg], track_n_vs_t=True))

# and finally for phi = 0.1
simulations_phi_zero_point_one = list()
while len(simulations_phi_zero_point_one) < 5:
    simulations_phi_zero_point_one.append(run_sim(t_phi_sim, 100000, {**standard_params, 'phi': [0.1, 0.1]}, n0=[N_avg, D - N_avg], track_n_vs_t=True))

# Plot everything
fig, ((ax_top_left, ax_top_right), (ax_bottom_left, ax_bottom_right)) = plt.subplots(2, 2, figsize=(6 * 1.2, 3.2 * 1.2))

# Top left panel: simulations of alpha
for i, simulation in enumerate(simulations_alpha_positive):
    label = "$α$ = 0.5" if i == 0 else None  # Only label once
    n_vs_t = simulation["n_vs_t"]
    ax_top_left.plot(n_vs_t[:,0],n_vs_t[:,1]+n_vs_t[:,2], color='#c23616', label=label, alpha=0.6)
for i, simulation in enumerate(simulations_alpha_negative):
    label = "$α$ = -0.5" if i == 0 else None  # Only label once
    n_vs_t = simulation["n_vs_t"]
    ax_top_left.plot(n_vs_t[:,0],n_vs_t[:,1]+n_vs_t[:,2], color='#10ac84', label=label, alpha=0.6)
ax_top_left.set_xlim(0, t_alpha_example_sim)
ax_top_left.set_xticks([0, t_alpha_example_sim])
ax_top_left.set_ylim(0, 60)
ax_top_left.set_yticks([0, 30, 60])
ax_top_left.set_ylabel("# of dividing cells $D$")
ax_top_left.legend()

# Bottom left panel: phi examples
for i, simulation in enumerate(simulations_phi_zero_point_nine):
    label = "$φ$ = 0.9" if i == 0 else None  # Only label once
    n_vs_t = simulation["n_vs_t"]
    ax_bottom_left.plot(n_vs_t[:, 0], n_vs_t[:, 1] + n_vs_t[:, 2], color='#ff9f43', label=label, alpha=0.6)
for i, simulation in enumerate(simulations_phi_zero_point_one):
    label = "$φ$ = 0.1" if i == 0 else None  # Only label once
    n_vs_t = simulation["n_vs_t"]
    ax_bottom_left.plot(n_vs_t[:, 0], n_vs_t[:, 1] + n_vs_t[:, 2], color='#576574', label=label, alpha=0.6)
ax_bottom_left.set_xlim(0, t_phi_sim)
ax_bottom_left.set_xticks([0, t_phi_sim])
ax_bottom_left.set_ylim(0, 60)
ax_bottom_left.set_yticks([0, 30, 60])
ax_bottom_left.legend()
ax_bottom_left.set_xlabel("Time (h)")
ax_bottom_left.set_ylabel("# of dividing cells $D$")

# Bottom right panel: depletion and overgrowth rates
ax_bottom_right.plot(phi_range, C, color='royalblue', label='_nolegend_')
ax_bottom_right.plot(phi_range, C, 'o', color='royalblue')
ax_bottom_right.plot(phi_range, E, color='violet', label='_nolegend_')
ax_bottom_right.plot(phi_range, E, 'o', color='violet')
ax_bottom_right.legend(['Depletion', 'Overgrowth'])
ax_bottom_right.set_ylabel('Rate (events/1,000h)')
ax_bottom_right.set_xlabel('$φ$')
ax_bottom_right.set_ylim([-0.05, 0.6])

# Top right panel: CoV
ax_top_right.plot(phi_range, div, 'o', color='#9D9D9C')
ax_top_right.plot(phi_range, div, color='#9D9D9C', label='_nolegend_')
ax_top_right.set_ylabel('Coefficient of variation')
ax_top_right.set_ylim([-.05, 1.05])
ax_top_right.xaxis.set_ticklabels([])

plt.tight_layout()
plt.show()
