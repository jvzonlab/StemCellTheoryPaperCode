import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
mpl.rcParams['pdf.fonttype'] = 42

from two_compartment_model_lib import run_sim

#%% set sweep parameters

np.random.seed(12)

plt.clf()

# fix total number of dividing cells
S = 10
D = 30
# fix cell cycle parameters (based on measured values)
T=[16.153070175438597,3.2357834505600382]

# total simulation time
t_sim=1e1

#maximum limit of number of dividing cells to stop simulation
n_max = 15*D

#define growth rate
alpha_n = 0


t_sim=1e3

t_lineage_range = [0,50]

plt.figure(1)

phi = 0.9
params = {'S':int(np.round(S)), 'alpha':[alpha_n, alpha_n], 'phi':[phi,phi], 'T':T }

p=(phi+alpha_n)/2
q=(phi-alpha_n)/2
r=1-p-q
print("p:%f, q:%f, r:%f"%(p,q,r))

for i in range(0,3):
    res = run_sim( t_sim, n_max, params, n0=[S,D-S], track_lineage_time_interval=t_lineage_range, track_n_vs_t=True)

    n_vs_t=res['n_vs_t']
    plt.plot(n_vs_t[:,0],n_vs_t[:,1]+n_vs_t[:,2],'-r')
res_asym = res

phi = 0.1
params = {'S':int(np.round(S)), 'alpha':[alpha_n, alpha_n], 'phi':[phi,phi], 'T':T }

p=(phi+alpha_n)/2
q=(phi-alpha_n)/2
r=1-p-q
print("p:%f, q:%f, r:%f"%(p,q,r))

for i in range(0,3):
    res = run_sim( t_sim, n_max, params, n0=[S,D-S], track_lineage_time_interval=t_lineage_range, track_n_vs_t=True)

    n_vs_t=res['n_vs_t']
    plt.plot(n_vs_t[:,0],n_vs_t[:,1]+n_vs_t[:,2],'-b')
res_sym = res

plt.xlabel('Time (hours)')
plt.ylabel('# of dividing cells')
    
#plt.subplot(122)
#
#L_list=res['Lineage'][0:10]
#x0=0
#random.seed(1)
#ra_L = random.choices(L_list, k = max_lin)
#for i in ra_L:
#    w = i.draw_lineage(t_lineage_range[1],x0,show_cell_id=False)
#    x0+=w+2
#plt.ylim([t_lineage_range[1],t_lineage_range[0]])
#plt.xlim([-2,x0])
#plt.xticks([])
#plt.yticks([])
#
