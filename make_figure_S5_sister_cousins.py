import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
import time
import seaborn as sns
import pandas as pd

from stem_cell_model.lineages import Lineages as Lineage, Lineages
import stem_cell_model.two_compartment_model_space as model

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'  # export text as text in SVG, not as paths

import matplotlib.colors as mcolors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = truncate_colormap(plt.get_cmap("magma_r"), 0, 0.75)

#%%

inch2cm=1/2.54
fig = plt.figure(1,figsize=(inch2cm*19,inch2cm*19))
plt.clf()

ax_A=[]
W=0.095
x_sp = 0.02
for i in [0,1]:
    ax_A.append( plt.axes([0.08+i*(W+x_sp),0.8,W,W]) )
ax_A_cb = plt.axes( [0.08+W+x_sp+0.01+W,0.8,0.015,W] )

ax_B=[]
for i in [0,1,2,3]:
    ax_B.append( plt.axes([0.51+i*(W+x_sp),0.8,W,W]) )
ax_B_cb = plt.axes( [0.51+3*(W+x_sp)+0.01+W,0.8,0.015,W] )

ax_C=[]
W=0.21
x_sp = 0.05
for n in [0,1]:
    ax_C.append( plt.axes([0.08+n*(W+x_sp),0.4,W,0.3]) )

#%% First, get experimental data for sister/cousin correlations

states = ['divided', 'did not divide']
states_lbl = ['divides', 'never divides']

# load and plot sister data
DF = pd.read_excel(r"Experimental data/SisterCousinDivisionBehavior_organoids.xlsx",sheet_name='Sisters',engine='openpyxl')
df = DF[0:3]
m = { i:n for (i,n) in enumerate( df['s2\s1'] )}
df = df.rename(index=m)
df = df[ states ].loc[states]

# get corresponding numpy array
S_expt = df.to_numpy()
# make triangle array, sums the off-diagonal elements, i.e. s_ij + s_ji, 
# reflect that there is no difference between S1 and S2
S_expt = np.tril(S_expt,0) + np.tril(S_expt.T,-1)
# finally, normalize
S_expt /= S_expt.sum()

mask = np.zeros( S_expt.shape, dtype=bool)
mask[ np.where(S_expt==0) ] = True
sns.heatmap(S_expt, ax=ax_A[0], annot=True, vmin = 0.03, vmax=0.41, mask=mask, cmap=cmap, cbar=False, xticklabels = states_lbl, yticklabels = states_lbl)

# load and plot cousin data
DF = pd.read_excel(r"Experimental data/SisterCousinDivisionBehavior_organoids.xlsx",sheet_name='Cousins',engine='openpyxl')
df1 = DF[0:3]
df2 = DF[7:10]
m = { i:n for (i,n) in enumerate( DF['c3(c4)\c1(c2)'] )}
df1 = df1.rename(index=m)
df2 = df2.rename(index=m)
df1 = df1[ states ].loc[states]
df2 = df2[ states ].loc[states]

C_expt = np.array( df1.to_numpy() + df2.to_numpy(), dtype=float )
C_expt = np.tril(C_expt,0) + np.tril(C_expt.T,-1)
C_expt /= C_expt.sum()

mask = np.zeros( C_expt.shape, dtype=bool)
mask[ np.where(C_expt==0) ] = True
sns.heatmap(C_expt, ax=ax_A[1], annot=True, vmin = 0.03, vmax=0.5, mask=mask, cmap=cmap, cbar=True, cbar_ax = ax_A_cb, cbar_kws ={'label':'Fraction'}, xticklabels = states_lbl, yticklabels = [])

for a in ax_A:
    a.spines['left'].set_visible(True)
    a.spines['bottom'].set_visible(True)
    a.tick_params(labelsize=7)

ax_A_cb.set_frame_on(True)
ax_A_cb.tick_params(direction='in',labelsize=6)

ax_A[1].set_xlabel('C2', fontsize=10)
ax_A[0].set_xlabel('S2', fontsize=10)
ax_A[0].set_ylabel('S1', fontsize=10)

#%%

D = 30
alpha_p = 0.9
alpha_d = -0.9
S = D / np.log(1+alpha_p)*alpha_d/(alpha_d - alpha_p)

t_sim=200
n_max=1e5
T=[16.153070175438597,3.2357834505600382]
params={'S':int(S), 'phi':[0.95,0.95], 'alpha':[alpha_p,alpha_d], 'T':T, 'a':10}

N_avg = params['alpha'][0]*params['S']
M_avg = np.log(1+params['alpha'][0])*params['S']*(params['alpha'][1]-params['alpha'][0])/params['alpha'][1] - params['alpha'][0]*params['S']

#%%

np.random.seed(0)

t_sim=100

a_list=[1 / T[0], 100 / T[0]]
L_ids = [ [5], [12, 17]]
for m in range(len(a_list)):

    params['a']=a_list[m]
    
    S = np.zeros( (2,2), dtype=float)
    C = np.zeros( (2,2), dtype=float)
    
    n0 = ( int(N_avg), int(M_avg) )
    data = model.run_sim_niche( t_sim,n_max, params, n0=n0, track_lineage_time_interval=[0,100], track_n_vs_t=True)
    lineages: Lineages = data['Lineage']

    x0=0
    plt.sca(ax_C[m])
    for l in L_ids[m]:
        W = lineages.draw_single_lineage(ax_C[m], lineages[l], t_sim, x0, col_default="#009CB5", col_comp_0="#7FB840")
        x0 += W

    ax_C[m].set_ylim([100,0])

    ax_C[m].tick_params(direction='in',labelsize=8)
    ax_C[m].set_xticks([])
    ax_C[m].set_title('rT=%1.1f' % (a_list[m]*T[0]) )
    
ax_C[0].set_ylabel('Time (h)')
    
#%% simulation for non-mixed niche

t_sim=200
a_list=[0.1 / T[0], 1 / T[0], 10 / T[0], 100 / T[0]]
for m in range(0,4):

    params['a']=a_list[m]
    
    S = np.zeros( (2,2), dtype=float)
    C = np.zeros( (2,2), dtype=float)
    
    n0 = ( int(N_avg), int(M_avg) )
    for n in range(0,1): #50):
        data = model.run_sim_niche( t_sim,n_max, params, n0=n0, track_lineage_time_interval=[100,t_sim], track_n_vs_t=False)
        
        # x_avg = data['Moments']['mean']/t_sim
        # x2_avg = data['Moments']['sq']/t_sim
        # sd = np.sqrt(x2_avg - x_avg**2)
        # print( "<N>, predicted:%f, sim: %f +/- %f" % (N_avg,x_avg[0],sd[0]))
        # print( "<M>, predicted:%f, sim: %f +/- %f" % (M_avg,x_avg[1],sd[1]))

        lineages = data['Lineage']
        # W = L.draw_lineage(t_sim,x0)
        # x0 += W
        stats = lineages.count_divisions()

        S[0,0] += stats.sisters_symmetric_non_dividing
        S[1,1] += stats.sisters_symmetric_dividing
        S[1,0] += stats.sisters_asymmetric

        C[0,0] += stats.cousins_symmetric_non_dividing
        C[1,1] += stats.cousins_symmetric_dividing
        C[1,0] += stats.cousins_asymmetric

    print(S.sum())
    S /= S.sum()
    C /= C.sum()
    print(C)
    
    
    # mask = np.zeros( S.shape, dtype=bool)
    # mask[ np.where(S==0) ] = True
    # sns.heatmap(S, vmin = 0.03, vmax=0.5, ax=ax_B[m], mask=mask, cmap=cmap, cbar=False, xticklabels = states_lbl, yticklabels = states_lbl)
    
    mask = np.zeros( C.shape, dtype=bool)
    mask[ np.where(C==0) ] = True
    if m==0:
        sns.heatmap(C, vmin = 0.03, vmax=0.5, ax=ax_B[m], annot=True, mask=mask, cmap=cmap, cbar=False, xticklabels = states_lbl, yticklabels = states_lbl)
    elif m==1 or m==2:
        sns.heatmap(C, vmin = 0.03, vmax=0.5, ax=ax_B[m], annot=True, mask=mask, cmap=cmap, cbar=False, xticklabels = states_lbl, yticklabels = [])
    else:
        sns.heatmap(C, vmin = 0.03, vmax=0.5, ax=ax_B[m], annot=True, mask=mask, cmap=cmap, cbar=True, cbar_ax = ax_B_cb, xticklabels = states_lbl, yticklabels = [])

    ax_B[m].set_title('aT=%1.1f' % (a_list[m] * T[0]))

    
for a in ax_B:
    a.spines['left'].set_visible(True)
    a.spines['bottom'].set_visible(True)
    a.tick_params(labelsize=7)

ax_B_cb.set_frame_on(True)
ax_B_cb.tick_params(direction='in',labelsize=6)

ax_B[2].set_xlabel('C2', fontsize=10)
ax_B[1].set_xlabel('C2', fontsize=10)
ax_B[0].set_xlabel('C2', fontsize=10)
ax_B[0].set_ylabel('C1', fontsize=10)


print(C_expt)
plt.show()
