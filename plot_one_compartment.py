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

# First for alpha = {-1, 1}, phi = 1
t_alpha_example_sim = 50
np.random.seed(11)
simulation_alpha_negative = run_sim(t_alpha_example_sim, 100000, {**standard_params, 'alpha': [-0.5, -0.5]}, n0=[N_avg, D - N_avg], track_n_vs_t=True)
np.random.seed(2)
simulation_alpha_positive = run_sim(t_alpha_example_sim, 100000, {**standard_params, 'alpha': [0.5, 0.5]}, n0=[N_avg, D - N_avg], track_n_vs_t=True)

# Then longer-term
t_phi_sim = 1000
simulations_for_phi = list()
seeds = [4, 17, 10]
for phi in [0.9, 0.9, 0.1]:
    np.random.seed(seeds.pop())
    results = run_sim(t_phi_sim, 100000, {**standard_params, 'phi': [phi, phi]}, n0=[N_avg, D - N_avg], track_n_vs_t=True)
    results["phi"] = phi
    simulations_for_phi.append(results)


# Plot everything
fig, ((ax_top_left, ax_top_right), (ax_bottom_left, ax_bottom_right)) = plt.subplots(2, 2, figsize=(6 * 1.2, 3.2 * 1.2))

# Top left panel: simulations of alpha
n_vs_t = simulation_alpha_positive["n_vs_t"]
ax_top_left.plot(n_vs_t[:,0],n_vs_t[:,1]+n_vs_t[:,2], color='#ff9f43', label="$α$ = 0.5")
n_vs_t = simulation_alpha_negative["n_vs_t"]
ax_top_left.plot(n_vs_t[:,0],n_vs_t[:,1]+n_vs_t[:,2], color='#576574', label="$α$ = -0.5")

ax_top_left.set_xlim(0, t_alpha_example_sim)
ax_top_left.set_xticks([0, t_alpha_example_sim])
ax_top_left.set_ylim(0, 60)
ax_top_left.set_yticks([0, 30, 60])
ax_top_left.set_ylabel("# of dividing cells $D$")
ax_top_left.legend()

# Bottom left panel: phi examples
colors = ["#c23616", "#c23616", "#10ac84"]
labels = ["$φ$ = 0.9", None, "$φ$ = 0.1"]
for simulation_results, color, label in zip(simulations_for_phi, colors, labels):
    n_vs_t = simulation_results["n_vs_t"]
    phi = simulation_results["phi"]
    ax_bottom_left.plot(n_vs_t[:, 0], n_vs_t[:, 1] + n_vs_t[:, 2], color=color, label=label)
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
