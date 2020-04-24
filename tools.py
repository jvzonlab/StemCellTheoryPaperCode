import numpy as np

def get_statistics(run_data):
    n_mean = run_data['mean']/run_data['t_tot']
    n_std_sq = run_data['sq']/run_data['t_tot'] - n_mean**2
    cc_NM = run_data['prod']/run_data['t_tot'] - n_mean[0]*n_mean[1]
    D_std_sq = n_std_sq[0]+n_std_sq[1]+2*cc_NM
    
    d_mean = sum(n_mean)
    
    n_std=np.sqrt( n_std_sq[0] )
    m_std=np.sqrt( n_std_sq[1] )
    
    n_coeff_var = n_std/n_mean[0]
    m_coeff_var = m_std/n_mean[1]
    
    if abs(D_std_sq) < 0.001: #to correct precision errors (eg D is -6.59659826763e-13 instead of 0)
        D_std_sq = 0
    D_std=np.sqrt( D_std_sq )
    
    d_coeff_var = D_std/d_mean

    f_collapse=1000*run_data['n_runs_ended_early']/run_data['t_tot'] #rate: event per 1,000 h

    #average time per simulation
    if run_data['n_runs_ended_early'] == 0:
        f_collapse_t = run_data['t_tot']
    else: 
        f_collapse_t = run_data['t_tot']/run_data['n_runs_ended_early']
    
    if 'n_explosions' in run_data.keys():
#        n_explosions=run_data['n_explosions'] #event number
        n_explosions=1000*run_data['n_explosions']/run_data['t_tot'] #rate: event per 1,000 h
        if run_data['n_explosions'] == 0:
            n_explosions_t=run_data['t_tot']
        else:
            n_explosions_t=run_data['t_tot']/run_data['n_explosions']
        return n_std, m_std, D_std,f_collapse,n_explosions,f_collapse_t,n_explosions_t,n_mean, d_mean, n_coeff_var,m_coeff_var,d_coeff_var
    else:
        return n_std, m_std, D_std,f_collapse,f_collapse_t,n_mean, d_mean, n_coeff_var, m_coeff_var,d_coeff_var
    
    
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
    
    for s in sim_data: 
        sweep_param = s[0]
        run_data = s[1]
        if sweep_param['phi'][0]==phi:
            n_std, m_std, D_std,f_collapse,f_collapse_t,n_mean, d_mean, n_cv, m_cv, d_cv = get_statistics(run_data)
            alpha=sweep_param['alpha']
            # find indeces i and j corresponding to the current parameters alpha_n,m        
            i = np.where(np.abs(alpha[0]-alpha_n_range)==np.abs((alpha[0]-alpha_n_range)).min())[0][0]
            j = np.where(np.abs(alpha[1]-alpha_m_range)==np.abs((alpha[1]-alpha_m_range)).min())[0][0]
        
            N[i,j] = n_std
            M[i,j] = m_std
            D[i,j] = D_std
            C[i,j] = f_collapse
            C_t[i,j] = f_collapse_t
            MEAN[i,j] = d_mean
            N_CV[i,j] = n_cv
            M_CV[i,j] = m_cv
            D_CV[i,j] = d_cv
    
    return N,M,D,C,C_t,MEAN,N_CV,M_CV,D_CV

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
            
            n_std, m_std, D_std,f_collapse,f_collapse_t,n_mean, d_mean, n_cv, m_cv, d_cv = get_statistics(run_data)
            alpha = sweep_param['alpha']
            phi = sweep_param['phi']

            # find indeces i and j corresponding to the current parameters    
            i = np.where(np.abs(phi[0]-phi_range)==np.abs((phi[0]-phi_range)).min())[0][0]
            j = np.where(np.abs(alpha[0]-alpha_n_range)==np.abs((alpha[0]-alpha_n_range)).min())[0][0]
            
            N[i,j] = n_std
            M[i,j] = m_std
            D[i,j] = D_std
            C[i,j] = f_collapse
            C_t[i,j] = f_collapse_t
            MEAN[i,j] = d_mean
            D_CV[i,j] = d_cv
            N_CV[i,j] = n_cv
            M_CV[i,j] = m_cv
    
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
    
    for s in sim_data: 
        sweep_param = s[0]
        run_data = s[1]
        
        alpha=sweep_param['alpha']
        phi=sweep_param['phi'][0]
        
        if np.abs(alpha[0]+alpha[1])<1e-3:
            
            n_std, m_std, D_std,f_collapse,f_collapse_t,n_mean, d_mean, n_cv, m_cv, d_cv = get_statistics(run_data)
            alpha = sweep_param['alpha']
            phi = sweep_param['phi']

            # find indeces i and j corresponding to the current parameters    
            i = np.where(np.abs(phi[0]-phi_range)==np.abs((phi[0]-phi_range)).min())[0][0]
            j = np.where(np.abs(alpha[0]-alpha_n_range)==np.abs((alpha[0]-alpha_n_range)).min())[0][0]
            
            N[i,j] = n_std
            M[i,j] = m_std
            D[i,j] = D_std
            C[i,j] = f_collapse
            C_t[i,j] = f_collapse_t
            MEAN[i,j] = d_mean
            D_CV[i,j] = d_cv
            N_CV[i,j] = n_cv
            M_CV[i,j] = m_cv
    
    return N,M,D,C,C_t,MEAN,N_CV,M_CV,D_CV