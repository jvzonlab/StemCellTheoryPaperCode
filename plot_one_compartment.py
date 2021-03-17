
import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
from stem_cell_model import tools

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

    n,m,d,c,e,c_t,e_t,mean,d_mean,n_coeff_var,m_coeff_var,d_coeff_var = tools.get_statistics(run_data)
    
    # get index i for current parameter phi
    i = np.where(phi_range == phi)[0][0]

    N[i] = n
    M[i] = m
    D[i] = d
    C[i] = c
    E[i] = e
    C_t[i] = c_t
    E_t[i] = e_t
    MEAN[i] = d_mean
    div[i] = d_coeff_var

#%%

fig, (ax2,ax1) = plt.subplots(2, 1, figsize=(5, 5))  

ax1.plot(phi_range,C, color='royalblue', label='_nolegend_')
ax1.plot(phi_range,C,'o', color='royalblue')
ax1.plot(phi_range,E, color='violet', label='_nolegend_')
ax1.plot(phi_range,E, 'o', color='violet')
ax1.legend(['Depletion','Overgrowth'])
ax1.set_ylabel('Rate (events/1,000h)')
ax1.set_xlabel('phi')
ax1.set_ylim([-0.05,0.6])

ax2.plot(phi_range,div,'o', color='yellowgreen')
ax2.plot(phi_range,div, color='yellowgreen', label='_nolegend_')
ax2.set_ylabel('Coefficient of variation')
ax2.set_ylim([-.05,1.05])
ax2.xaxis.set_ticklabels([])
