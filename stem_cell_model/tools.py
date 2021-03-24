from typing import Dict, Any, Optional

import numpy as np


class SingleRunStatistics:
    """Statistics of a single simulation run."""

    n_mean: float  # Mean of the number of proliferating cells in the niche compartment
    m_mean: float  # Mean of the number of proliferating cells in the differentiation compartment
    d_mean: float  # Mean number of dividing cells in the entire model.

    n_std: float  # Standard deviation of the number of proliferating cells in the niche compartment
    m_std: float  # Standard deviation of the number of proliferating cells in the differentiation compartment
    d_std: float  # Standard deviation of the total number of proliferating cells

    f_collapse: float  # Number of collapses per 1000 simulation time units
    f_collapse_t: float  # Total simulation time divided by the number of collapses
    n_explosions: Optional[float] = None  # Number of explosions per 1000 simulation time units
    n_explosions_t: Optional[float] = None  # Total simulation time divided by the number of explosions

    n_coeff_var: float  # Variation coefficient of the number of proliferating cells in the niche compartment
    m_coeff_var: float  # Variation coefficient of the number of proliferating cells in the differentiation compartment
    d_coeff_var: float  # Variation coefficient of the number of proliferating cells in both compartments


def get_single_run_statistics(run_data: Dict[str, Any]) -> SingleRunStatistics:
    """Gets the statistics of a single simulation run."""
    out = SingleRunStatistics()

    n_m_mean = run_data['mean'] / run_data['t_tot']
    out.n_mean = n_m_mean[0]
    out.m_mean = n_m_mean[1]

    n_std_sq = run_data['sq'] / run_data['t_tot'] - n_m_mean ** 2
    cc_NM = run_data['prod'] / run_data['t_tot'] - out.n_mean * out.m_mean
    D_std_sq = n_std_sq[0] + n_std_sq[1] + 2 * cc_NM

    d_mean = sum(n_m_mean)

    out.n_std = np.sqrt(n_std_sq[0])
    out.m_std = np.sqrt(n_std_sq[1])

    out.n_coeff_var = out.n_std / out.n_mean
    out.m_coeff_var = out.m_std / out.m_mean

    if abs(D_std_sq) < 0.001:  # to correct precision errors (eg D is -6.59659826763e-13 instead of 0)
        D_std_sq = 0
    out.d_std = np.sqrt(D_std_sq)

    out.d_coeff_var = out.d_std / d_mean

    out.f_collapse = 1000 * run_data['n_runs_ended_early'] / run_data['t_tot']  # rate: event per 1,000 h

    # average time per simulation
    if run_data['n_runs_ended_early'] == 0:
        out.f_collapse_t = run_data['t_tot']
    else:
        out.f_collapse_t = run_data['t_tot'] / run_data['n_runs_ended_early']

    if 'n_explosions' in run_data.keys():
        #        n_explosions=run_data['n_explosions'] #event number
        out.n_explosions = 1000 * run_data['n_explosions'] / run_data['t_tot']  # rate: event per 1,000 h
        if run_data['n_explosions'] == 0:
            out.n_explosions_t = run_data['t_tot']
        else:
            out.n_explosions_t = run_data['t_tot'] / run_data['n_explosions']
    return out


def plot_alphas_for_constant_phi(phi,sim_data,alpha_n_range,alpha_m_range,phi_range,Np):
    N = np.zeros((Np,Np))
    M = np.zeros((Np,Np))
    D = np.zeros((Np,Np))
    C = np.zeros((Np,Np))
    C_t = np.zeros((Np,Np))
    MEAN = np.zeros((Np,Np))
    N_CV = np.zeros((Np,Np))
    M_CV = np.zeros((Np,Np))
    D_CV = np.zeros((Np,Np))
    S = np.zeros((Np,Np))
    
    for s in sim_data: 
        sweep_param = s[0]
        run_data = s[1]
        if sweep_param['phi'][0]==phi:
            single_run_statistics = get_single_run_statistics(run_data)
            alpha=sweep_param['alpha']
            # find indeces i and j corresponding to the current parameters alpha_n,m        
            i = np.where(np.abs(alpha[0]-alpha_n_range)==np.abs((alpha[0]-alpha_n_range)).min())[0][0]
            j = np.where(np.abs(alpha[1]-alpha_m_range)==np.abs((alpha[1]-alpha_m_range)).min())[0][0]
        
            N[i,j] = single_run_statistics.n_std
            M[i,j] = single_run_statistics.m_std
            D[i,j] = single_run_statistics.d_std
            C[i,j] = single_run_statistics.f_collapse
            C_t[i,j] = single_run_statistics.f_collapse_t
            MEAN[i,j] = single_run_statistics.d_mean
            N_CV[i,j] = single_run_statistics.n_coeff_var
            M_CV[i,j] = single_run_statistics.m_coeff_var
            D_CV[i,j] = single_run_statistics.d_coeff_var
            S[i,j] = sweep_param["S"]
    
    return N,M,D,C,C_t,MEAN,N_CV,M_CV,D_CV,S

def plot_alpha_n_vs_phi(alpha_m,sim_data,alpha_n_range,alpha_m_range,phi_range,Np):
    N = np.zeros((Np,Np))
    M = np.zeros((Np,Np))
    D = np.zeros((Np,Np))
    C = np.zeros((Np,Np))
    C_t = np.zeros((Np,Np))
    MEAN = np.zeros((Np,Np))
    N_CV = np.zeros((Np,Np))
    M_CV = np.zeros((Np,Np))
    D_CV = np.zeros((Np,Np))
    
    for s in sim_data: 
        sweep_param = s[0]
        run_data = s[1]

        if sweep_param['alpha'][1] - alpha_m < 0.001:
            
            single_run_statistics = get_single_run_statistics(run_data)
            alpha = sweep_param['alpha']
            phi = sweep_param['phi']

            # find indeces i and j corresponding to the current parameters    
            i = np.where(np.abs(phi[0]-phi_range)==np.abs((phi[0]-phi_range)).min())[0][0]
            j = np.where(np.abs(alpha[0]-alpha_n_range)==np.abs((alpha[0]-alpha_n_range)).min())[0][0]
            
            N[i,j] = single_run_statistics.n_std
            M[i,j] = single_run_statistics.m_std
            D[i,j] = single_run_statistics.d_std
            C[i,j] = single_run_statistics.f_collapse
            C_t[i,j] = single_run_statistics.f_collapse_t
            MEAN[i,j] = single_run_statistics.d_mean
            D_CV[i,j] = single_run_statistics.d_coeff_var
            N_CV[i,j] = single_run_statistics.n_coeff_var
            M_CV[i,j] = single_run_statistics.m_coeff_var
    
    return N,M,D,C,C_t,MEAN,N_CV,M_CV,D_CV

def plot_opposite_alphas(sim_data,alpha_n_range,alpha_m_range,phi_range,Np):
    N = np.zeros((Np,Np))
    M = np.zeros((Np,Np))
    D = np.zeros((Np,Np))
    C = np.zeros((Np,Np))-1
    C_t = np.zeros((Np,Np))
    MEAN = np.zeros((Np,Np))
    N_CV = np.zeros((Np,Np))
    M_CV = np.zeros((Np,Np))
    D_CV = np.zeros((Np,Np))
    S = np.zeros((Np,Np))
    
    for s in sim_data: 
        sweep_param = s[0]
        run_data = s[1]
        
        alpha=sweep_param['alpha']
        phi=sweep_param['phi'][0]
        
        if np.abs(alpha[0]+alpha[1])<1e-3:
            
            single_run_statistics = get_single_run_statistics(run_data)
            alpha = sweep_param['alpha']
            phi = sweep_param['phi']

            # find indeces i and j corresponding to the current parameters    
            i = np.where(np.abs(phi[0]-phi_range)==np.abs((phi[0]-phi_range)).min())[0][0]
            j = np.where(np.abs(alpha[0]-alpha_n_range)==np.abs((alpha[0]-alpha_n_range)).min())[0][0]
            
            N[i,j] = single_run_statistics.n_std
            M[i,j] = single_run_statistics.m_std
            D[i,j] = single_run_statistics.d_std
            C[i,j] = single_run_statistics.f_collapse
            C_t[i,j] = single_run_statistics.f_collapse_t
            MEAN[i,j] = single_run_statistics.d_mean
            D_CV[i,j] = single_run_statistics.d_coeff_var
            N_CV[i,j] = single_run_statistics.n_coeff_var
            M_CV[i,j] = single_run_statistics.m_coeff_var
            S[i,j] = sweep_param["S"]
    
    return N,M,D,C,C_t,MEAN,N_CV,M_CV,D_CV,S
