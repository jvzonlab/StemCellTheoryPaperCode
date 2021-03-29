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
np.random.seed(1)
standard_params = {'S':int(np.round(S)), 'phi':[1, 1], 'T':T, 'alpha': [0, 0]}

# First for alpha = {-1, 1}, phi = 1
t_alpha_example_sim = 40
simulation_alpha_minus_one = run_sim(t_alpha_example_sim, 100000, {**standard_params, 'alpha': [-1, -1]}, n0=[N_avg, D - N_avg], track_n_vs_t=True)
simulation_alpha_one = run_sim(t_alpha_example_sim, 100000, {**standard_params, 'alpha': [1, 1]}, n0=[N_avg, D - N_avg], track_n_vs_t=True)

# Then for phi from 0 <= phi <= 1
t_phi_sim = 1000
simulation_by_phi = dict()
seeds = [2, 55, 4]
for phi in [0.0, 0.1, 1.0]:
    np.random.seed(seeds.pop())
    simulation_by_phi[float(phi)] = run_sim(t_phi_sim, 100000, {**standard_params, 'phi': [phi, phi]}, n0=[N_avg, D - N_avg], track_n_vs_t=True)


# Plot everything
fig, ((ax_top_left, ax_top_right), (ax_bottom_left, ax_bottom_right)) = plt.subplots(2, 2, figsize=(6, 4))

# Top left panel: simulations of alpha
n_vs_t = simulation_alpha_one["n_vs_t"]
ax_top_left.plot(n_vs_t[:,0],n_vs_t[:,1]+n_vs_t[:,2], color='violet', label="α = 1")
n_vs_t = simulation_alpha_minus_one["n_vs_t"]
ax_top_left.plot(n_vs_t[:,0],n_vs_t[:,1]+n_vs_t[:,2], color='royalblue', label="α = -1")

ax_top_left.set_xticks([])
ax_top_left.set_xlabel("Time")
ax_top_left.set_ylabel("# of dividing cells $D$")
ax_top_left.legend()

# Bottom left panel: phi examples
colors = ["violet", "royalblue", "yellowgreen", "#c23616"]
for phi, simulation_results in simulation_by_phi.items():
    n_vs_t = simulation_results["n_vs_t"]
    ax_bottom_left.plot(n_vs_t[:, 0], n_vs_t[:, 1] + n_vs_t[:, 2], color=colors.pop(), label=f"φ = {phi}")
ax_bottom_left.set_ylim(-5, 55)
ax_bottom_left.legend()
ax_bottom_left.set_xticks([])
ax_bottom_left.set_xlabel("Time")
ax_bottom_left.set_ylabel("# of dividing cells $D$")

# Bottom right panel: depletion and overgrowth rates
ax_bottom_right.plot(phi_range, C, color='royalblue', label='_nolegend_')
ax_bottom_right.plot(phi_range, C, 'o', color='royalblue')
ax_bottom_right.plot(phi_range, E, color='violet', label='_nolegend_')
ax_bottom_right.plot(phi_range, E, 'o', color='violet')
ax_bottom_right.legend(['Depletion', 'Overgrowth'])
ax_bottom_right.set_ylabel('Rate (events/1,000h)')
ax_bottom_right.set_xlabel('phi')
ax_bottom_right.set_ylim([-0.05, 0.6])

# Top right panel: CoV
ax_top_right.plot(phi_range, div, 'o', color='yellowgreen')
ax_top_right.plot(phi_range, div, color='yellowgreen', label='_nolegend_')
ax_top_right.set_ylabel('Coefficient of variation')
ax_top_right.set_ylim([-.05, 1.05])
ax_top_right.xaxis.set_ticklabels([])

plt.tight_layout()
plt.show()
