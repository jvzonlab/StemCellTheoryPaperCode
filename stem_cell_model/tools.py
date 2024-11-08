from typing import Dict, Any, Optional

import numpy as np
from numpy import ndarray

from stem_cell_model.results import MultiRunStats


class SingleParameterSetStatistics:
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


class ResultImages:
    N: ndarray
    M: ndarray
    D: ndarray
    C: ndarray
    C_t: ndarray
    MEAN: ndarray
    N_CV: ndarray
    M_CV: ndarray
    D_CV: ndarray
    S: ndarray
    n_explosions: ndarray
    n_explosions_t: ndarray

    def __init__(self, Np):
        self.N = np.zeros((Np, Np))
        self.M = np.zeros((Np, Np))
        self.D = np.zeros((Np, Np))
        self.C = np.zeros((Np, Np))
        self.C_t = np.zeros((Np, Np))
        self.MEAN = np.zeros((Np, Np))
        self.N_CV = np.zeros((Np, Np))
        self.M_CV = np.zeros((Np, Np))
        self.D_CV = np.zeros((Np, Np))
        self.S = np.zeros((Np, Np))
        self.n_explosions = np.zeros((Np, Np))
        self.n_explosions_t = np.zeros((Np, Np))


def get_single_parameter_set_statistics(run_data: MultiRunStats) -> SingleParameterSetStatistics:
    """Gets the statistics of the runs of a single set of simulation parameters."""
    out = SingleParameterSetStatistics()

    n_m_mean = run_data.nm_mean / run_data.t_tot
    out.n_mean = n_m_mean[0]
    out.m_mean = n_m_mean[1]

    n_std_sq = run_data.nm_sq / run_data.t_tot - n_m_mean ** 2
    cc_NM = run_data.nm_prod / run_data.t_tot - out.n_mean * out.m_mean
    D_std_sq = n_std_sq[0] + n_std_sq[1] + 2 * cc_NM

    out.d_mean = sum(n_m_mean)

    out.n_std = np.sqrt(n_std_sq[0])
    out.m_std = np.sqrt(n_std_sq[1])

    out.n_coeff_var = out.n_std / out.n_mean
    out.m_coeff_var = out.m_std / out.m_mean

    if abs(D_std_sq) < 0.001:  # to correct precision errors (eg D is -6.59659826763e-13 instead of 0)
        D_std_sq = 0
    out.d_std = np.sqrt(D_std_sq)

    out.d_coeff_var = out.d_std / out.d_mean

    out.f_collapse = 1000 * run_data.n_runs_ended_early / run_data.t_tot  # rate: event per 1,000 h

    # average time per simulation
    if run_data.n_runs_ended_early == 0:
        out.f_collapse_t = run_data.t_tot
    else:
        out.f_collapse_t = run_data.t_tot / run_data.n_runs_ended_early

    if run_data.n_explosions is not None:
        #        n_explosions=run_data['n_explosions'] #event number
        out.n_explosions = 1000 * run_data.n_explosions / run_data.t_tot  # rate: event per 1,000 h
        if run_data.n_explosions == 0:
            out.n_explosions_t = run_data.t_tot
        else:
            out.n_explosions_t = run_data.t_tot / run_data.n_explosions
    return out


def plot_alphas_for_constant_phi(phi,sim_data,alpha_n_range,alpha_m_range,phi_range,Np):
    out = ResultImages(Np)
    
    for s in sim_data: 
        sweep_param = s[0]
        run_data = MultiRunStats.from_dict(s[1])
        if sweep_param['phi'][0]==phi:
            statistics = get_single_parameter_set_statistics(run_data)
            alpha=sweep_param['alpha']
            # find indeces i and j corresponding to the current parameters alpha_n,m        
            i = np.where(np.abs(alpha[0]-alpha_n_range)==np.abs((alpha[0]-alpha_n_range)).min())[0][0]
            j = np.where(np.abs(alpha[1]-alpha_m_range)==np.abs((alpha[1]-alpha_m_range)).min())[0][0]
        
            out.N[i,j] = statistics.n_std
            out.M[i,j] = statistics.m_std
            out.D[i,j] = statistics.d_std
            out.C[i,j] = statistics.f_collapse
            out.C_t[i,j] = statistics.f_collapse_t
            out.MEAN[i,j] = statistics.d_mean
            out.N_CV[i,j] = statistics.n_coeff_var
            out.M_CV[i,j] = statistics.m_coeff_var
            out.D_CV[i,j] = statistics.d_coeff_var
            out.S[i,j] = sweep_param["S"]
            out.n_explosions[i, j] = statistics.n_explosions
            out.n_explosions_t[i, j] = statistics.n_explosions_t
    
    return out

def plot_alpha_n_vs_phi(alpha_m,sim_data,alpha_n_range,alpha_m_range,phi_range,Np):
    out = ResultImages(Np)
    
    for s in sim_data: 
        sweep_param = s[0]
        run_data = MultiRunStats.from_dict(s[1])

        if sweep_param['alpha'][1] - alpha_m < 0.001:
            
            statistics = get_single_parameter_set_statistics(run_data)
            alpha = sweep_param['alpha']
            phi = sweep_param['phi']

            # find indeces i and j corresponding to the current parameters    
            i = np.where(np.abs(phi[0]-phi_range)==np.abs((phi[0]-phi_range)).min())[0][0]
            j = np.where(np.abs(alpha[0]-alpha_n_range)==np.abs((alpha[0]-alpha_n_range)).min())[0][0]
            
            out.N[i,j] = statistics.n_std
            out.M[i,j] = statistics.m_std
            out.D[i,j] = statistics.d_std
            out.C[i,j] = statistics.f_collapse
            out.C_t[i,j] = statistics.f_collapse_t
            out.MEAN[i,j] = statistics.d_mean
            out.D_CV[i,j] = statistics.d_coeff_var
            out.N_CV[i,j] = statistics.n_coeff_var
            out.M_CV[i,j] = statistics.m_coeff_var
            out.S[i,j] = sweep_param["S"]
            out.n_explosions[i, j] = statistics.n_explosions
            out.n_explosions_t[i, j] = statistics.n_explosions_t
    
    return out

def plot_opposite_alphas(sim_data,alpha_n_range,alpha_m_range,phi_range,Np) -> ResultImages:
    out = ResultImages(Np)
    
    for s in sim_data: 
        sweep_param = s[0]
        run_data = MultiRunStats.from_dict(s[1])
        
        alpha=sweep_param['alpha']
        phi=sweep_param['phi'][0]
        
        if np.abs(alpha[0]+alpha[1])<1e-3:
            
            statistics = get_single_parameter_set_statistics(run_data)
            alpha = sweep_param['alpha']
            phi = sweep_param['phi']

            # find indeces i and j corresponding to the current parameters    
            i = np.where(np.abs(phi[0]-phi_range)==np.abs((phi[0]-phi_range)).min())[0][0]
            j = np.where(np.abs(alpha[0]-alpha_n_range)==np.abs((alpha[0]-alpha_n_range)).min())[0][0]
            
            out.N[i,j] = statistics.n_std
            out.M[i,j] = statistics.m_std
            out.D[i,j] = statistics.d_std
            out.C[i,j] = statistics.f_collapse
            out.C_t[i,j] = statistics.f_collapse_t
            out.MEAN[i,j] = statistics.d_mean
            out.D_CV[i,j] = statistics.d_coeff_var
            out.N_CV[i,j] = statistics.n_coeff_var
            out.M_CV[i,j] = statistics.m_coeff_var
            out.S[i,j] = sweep_param["S"]
            out.n_explosions[i, j] = statistics.n_explosions
            out.n_explosions_t[i, j] = statistics.n_explosions_t
    
    return out
