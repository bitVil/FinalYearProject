# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 12:50:12 2021

@author: David Fox
"""
import numpy as np
import ReservoirComputer as res
import ReservoirComputer_3 as ReCom
from scipy.signal import argrelextrema
import scipy.sparse as sp
from scipy.spatial import distance


def cont(RC, x0, rho_space, dt, T, store, t_listen):
    x0 = x0.copy()
    n = int(round(T/dt))
    t_points = np.linspace(0, T, n+1)
    pert_traj_pred = {}
    pert_traj_res = {}
    
    for rho in rho_space:
        # Train RC with new rho value
        RC.rho = rho
        RC.train(t_listen, method=res.RungeKutta4)
    
        # Start PR from previous PR traj endpoint,
        PR_traj, R3_traj = res.RungeKutta4(RC.PR, t_points).solve(x0)
        x0 = PR_traj[-2]
        pert_traj_res[rho] = PR_traj[-store:-2]
        pert_traj_pred[rho] = R3_traj[-store:-2]
        
    
    return pert_traj_pred, pert_traj_res


def loc_extrema(traj, rho_space):
    maxima = []
    maxima_ind = []
    minima = []
    minima_ind = []
    
    
    for i in range(len(rho_space)):
        z = traj[i][:,2]

        # Find local maxima
        max_ind = argrelextrema(z, np.greater)
        max_vals = z[max_ind]

        # Find local minima
        min_ind = argrelextrema(z, np.less)
        min_vals = z[min_ind]
    
        maxima.append(max_vals)
        minima.append(min_vals)
        maxima_ind.append(max_ind)
        minima_ind.append(min_ind)
        
    return maxima, minima, maxima_ind, minima_ind


def evolve(RC, ics, dt, T):
    ics.append(RC.r_T)
    n = int(round(T/dt))
    t_points = np.linspace(0, T, n+1)
    traj = []
    
    for r0 in ics:
        PR_traj, R3_traj = res.RungeKutta4(RC.PR, t_points).solve(r0)
        traj.append(R3_traj[0:-2])
    
    return traj


def angle(u, v):
    u_hat = u/np.linalg.norm(u)
    v_hat = v/np.linalg.norm(v)
    return np.arccos(np.clip(np.dot(u_hat, v_hat), -1.0, 1.0))


def gram_schmidt(v, tol=10e-16):
    e = np.zeros(v.shape)
    beta = np.zeros(v.shape[1])
    
    for i in range(v.shape[1]):
        u = v[:,i] - np.asarray([np.dot(v[:,i], e[:,j]) * e[:,j] for j in range(i)]).sum(axis=0)
        beta[i] = np.linalg.norm(u)
        
        if beta[i] < tol:
            e[:,i] = np.zeros(v.shape[0])
        else:
            e[:,i] = u/beta[i]
    
    return e, beta


def PR_lyap_RK4(f, h, n, x0, RC, W, max_mag=10e+150):
    x = np.zeros((n, x0.size))
    x[0] = x0
    n_last = n
    
    for i in range(n-1):
        k1 = h*f(x[i], RC, W)
        k2 = h*f(x[i] + 0.5*k1, RC, W)
        k3 = h*f(x[i] + 0.5*k2, RC, W)
        k4 = h*f(x[i] + k3, RC, W)
        x[i+1] = x[i] + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        delta_norm = np.linalg.norm(x[i+1,RC.N:])
        
        if delta_norm > max_mag or delta_norm < 1/max_mag:
            x = x[0:i+1]
            n_last = i
            break
    
    return x, n_last

def LR_lyap_RK4(f, h, n, x0, t0, RC, tol=10e+150):
    data = RC.data
    x = np.zeros((n, x0.size))
    x[0] = x0
    n_last = n
    
    for i in range(t0, min(t0 + n-1, data.size)):
        k1 = h*f(x[i], RC, data[i+1])
        k2 = h*f(x[i] + 0.5*k1, RC, data[i+1])
        k3 = h*f(x[i] + 0.5*k2, RC, data[i+1])
        k4 = h*f(x[i] + k3, RC, data[i+1])
        x[i+1] = x[i] + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        delta_norm = np.linalg.norm(x[i+1,RC.N:])
        
        if  delta_norm > tol or delta_norm < 1/tol:
            x = x[0:i+1]
            n_last = i
            break
    
    return x, n_last


def jac_LR(r, RC, drive):
    gamma, sigma, M, W_in = RC.gamma, RC.sigma, RC.M, RC.W_in
    a = 1/np.cosh(M.dot(r) + sigma*W_in.dot(drive))**2 
    M1 = sp.identity(r.size, format='csr').multiply(a).tocsr()
    M2 = M
    I = (sp.identity(r.size, format='csr'))

    return gamma*(-I + M1*M2)


def jac_PR(r, RC, W):
    gamma, sigma, M, N = RC.gamma, RC.sigma, RC.M, RC.N
    q_r = RC.PR.q(r)
    M1 = 1/np.cosh(M.dot(r) + sigma*W.dot(q_r))**2 
    DGH_r = sp.identity(N, format='csr').multiply(M1).tocsr()
    DH_r = M + sigma*(W[:,:N] + 2*W[:,N:]*r)
    I = (sp.identity(N, format='csr'))

    return gamma*(-I + DGH_r * DH_r)


def lyap_syst_LR(u, RC, drive):
    r = u[:int(u.size/2)]
    d = u[int(u.size/2):]
    
    r_next, est = RC.LR(r, drive, return_est=True)
    d = jac_LR(r, RC, drive).dot(d)
    return np.append(r_next, d) 


def lyap_syst_PR(u, RC, W):
    r = u[:int(u.size/2)]
    d = u[int(u.size/2):]
    
    r_next, est = RC.PR(r, return_est=True)
    d = jac_PR(r, RC, W).dot(d)
    return np.append(r_next, d) 


def lyap_spect_PR(RC, x0, q, h, t_ons, max_iters):
    N = x0.size # System dimension.
    p = q.shape[1] # How many exponents to find == how many vectors in columns of q.
    log_vols = np.zeros((max_iters, p)) # Matrix for storing p-piped log volumes.
    lyap_exps = np.zeros((max_iters, p)) # Rows are values of LE approximation for each step.
    W = RC.W_in*RC.W_out
    
    for j in range(max_iters):
        for i in range(p):
            u0 = np.append(x0, q[:,i])
            u0 = PR_lyap_RK4(lyap_syst_PR, h, t_ons, u0, RC, W)[0][-1]
            q[:,i] = u0[N:2*N]
            
        x0 = u0[:N]
        q, beta = gram_schmidt(q)
        
        if j >= 1:
            log_vols[j] = [1/(h*t_ons)*np.log(beta[:i].prod()) for i in range(1, p+1)]

            for i in range(p):
                if i == 0:
                    lyap_exps[j, i] = log_vols[:j,i].mean()
                else:
                    lyap_exps[j, i] = (log_vols[:j,i] - log_vols[:j,i-1]).mean()

    return lyap_exps, log_vols


def mcle(RC, x0, d0, h, t_t, max_iters, tol):
    A = np.asarray([])
    d = np.asarray([])
    N = x0.size
    t0 = 0
    delta = np.inf
    
    # Evolve coupled system to decay transience.
    u0 = np.append(x0, d0)
    u0, t0 = LR_lyap_RK4(lyap_syst_LR, h, t_t, u0, t0, RC)
    u0 = u0[-1]
    u0[N:] *= 1/np.linalg.norm(u0[N:])

    
    while len(A) <= 1 or delta > tol:
        u, t0 = LR_lyap_RK4(lyap_syst_LR, h, max_iters, u0, t0, RC)
        d = np.append(d, u[:, N:])
        A = np.append(A, 1/(h*t0)*np.log(np.linalg.norm(d[-1])))
        u0 = u[-1].copy()
        u0[N:] *= 1/np.linalg.norm(u0[N:])
        
        if (len(A) > 1):
            delta = np.abs(A.mean() - A[:-1].mean())
    
    d = np.asarray([np.linalg.norm(d_k) for d_k in d])
    mle = A.mean()
        
    return mle, A, d


def rho_sigma_orbits(rho_space, sigma_space, train_data, t_train, t_predict, t_listen, RC, hebb=False, eta=None, E=None):
    # Set up (rho, sigma) pairs to use as keys for dictionary of trajectories.
    rho, sigma = np.meshgrid(rho_space, sigma_space)
    param_space = np.array([rho.flatten(),sigma.flatten()]).T

    # Initialize dicts for trajectories.
    predicted_orbits = dict(keys=param_space)
    
    for rho, sigma in param_space:
        # Train Reservoir and make prediction for new parameter values.
        RC.sigma = sigma
        RC.rho = rho
        
        training_traj = RC.train(train_data, t_train, t_listen)
        if hebb == True:
            RC_heb = anti_heb(RC, train_data, t_train, t_listen, training_traj, eta, E)
            prediction = RC_heb.predict(t_predict)
        else:
            prediction = RC_heb.predict(t_predict)    

        # Store training trajectory and prediction in dicts
        predicted_orbits[(rho, sigma)] = prediction
        print((rho, sigma))
        
    #return training_orbits, predicted_orbits
    return predicted_orbits    


def is_fixed_point(x, tol, n):
    """
    x : (ndarray).
    n: (int) number of points to consider at end of x.
    tol: (float64) difference between any pair of last
         n points of x must be less than tol if fixed point.
    """
    dists = distance.cdist(x[-n:], x[-n:], 'euclidean')
    return np.max(dists) < tol


def is_limit_cycle(x, tol=1, n=3000):
    """
    x : (ndarray).
    n: (int) number of points to consider at end of x.
    tol: (float64) difference between any pair of last
         n points of x must be less than tol if fixed point.
    """
    
    maxima = []
    minima = []
    
    for i in range(x.shape[1]):


        z = x[-n:,i]

        # for local maxima
        max_indices = argrelextrema(z, np.greater)
        max_vals = z[max_indices]

        # for local minima
        min_indices = argrelextrema(z, np.less)
        min_vals = z[min_indices]

        maxima.append(max_vals)
        minima.append(min_vals)
        
    # First periodicity check, see if first differences are 0 in any coord.
    for i in range(x.shape[1]):
        if np.abs(np.max(np.absolute(np.diff(maxima[i], n=1)))) < tol:
            return True, [maxima, max_indices, minima, min_indices]
        
    return False, [maxima, max_indices, minima, min_indices]


def pred_RMSE(u, u_hat):
    T = max(u_hat.shape[0], u.shape[0])
    square_dist = np.asarray([np.linalg.norm(u_hat[t] - u[t])**2 for t in range(T)])
    return np.sqrt(square_dist.sum()/T)


def pred_NRMSE(u, u_hat, sd):
    T = max(u_hat.shape[0], u.shape[0])
    square_dist = np.asarray([np.linalg.norm(u_hat[t] - u[t])**2 for t in range(T)])
    return np.sqrt(square_dist.sum()/T)/sd


def bifmat(a_space, b_space, orbits, fp_tol=1e-6, lc_tol=1, fp_n=1000, lc_n=3000):
    
    a, b = np.meshgrid(a_space, b_space)
    param_space = np.array([a.flatten(),b.flatten()]).T
    attractor_type = dict(keys=param_space)
    
    for a, b in param_space:
        x = orbits[(a, b)]
        if is_fixed_point(x, fp_tol, fp_n):
            attractor_type[(a, b)] = 0
        elif is_limit_cycle(x, lc_tol, lc_n)[0]: 
            attractor_type[(a, b)] = 0.5
        else:
            attractor_type[(a, b)] = 1
            
    bifurcation_matrix = np.zeros((a_space.size, b_space.size))
    
    for i in range(a_space.size):
        for j in range(b_space.size):
            bifurcation_matrix[i, j] = attractor_type[(a_space[i], b_space[j])]
    
    return bifurcation_matrix


def RMSE_mat(a_space, b_space, orbits, val_data):
    
    a, b = np.meshgrid(a_space, b_space)
    param_space = np.array([a.flatten(),b.flatten()]).T
    RMSE = dict(keys=param_space)
    
    
    for a, b in param_space:
        x = orbits[(a, b)]
        RMSE[(a, b)] = pred_RMSE(x, val_data)
            
    bifurcation_matrix = np.zeros((a_space.size, b_space.size))
    
    for i in range(a_space.size):
        for j in range(b_space.size):
            bifurcation_matrix[i, j] = RMSE[(a_space[i], b_space[j])]
    
    return bifurcation_matrix


def RK4_step(x, dt, f, args):
    k_1 = f(x, args)
    k_2 = f(x + dt*k_1/2, args)
    k_3 = f(x + dt*k_2/2, args)
    k_4 = f(x + dt*k_3, args)
    return 1/6*dt*(k_1 + 2*k_2 + 2*k_3 + k_4)


def TPE(u_hat, dt, f, args):
    # Calculate approx. and ideal movement vectors.
    delta_hat = np.diff(u_hat, axis=0)
    delta = np.asarray([RK4_step(u_hat[i], dt, f, args) for i in range(u_hat.shape[0] - 1)])
    
    # Calculate norms of ideal movement vectors approx. movement vector error.
    delta_norm = np.linalg.norm(delta, axis=1)
    error_norm = np.linalg.norm(delta_hat - delta, axis=1)
    
    # Return TPE.
    return 1/(dt*u_hat.shape[0])*np.sum(np.multiply(error_norm, 1/delta_norm))


def TPE_mat(a_space, b_space, orbits, dt, f, args):
    
    a, b = np.meshgrid(a_space, b_space)
    param_space = np.array([a.flatten(),b.flatten()]).T
    RMSE = dict(keys=param_space)
    
    
    for a, b in param_space:
        x = orbits[(a, b)]
        RMSE[(a, b)] = TPE(x, dt, f, args)
            
    bifurcation_matrix = np.zeros((a_space.size, b_space.size))
    
    for i in range(a_space.size):
        for j in range(b_space.size):
            bifurcation_matrix[i, j] = RMSE[(a_space[i], b_space[j])]
    
    return bifurcation_matrix

def anti_heb(RC, train_data, t_points_training, t_listen, training_traj, eta, E, size_E=9999):
    x = training_traj
    X = np.zeros((RC.N,RC.N))
    M = RC.M.copy()

    for e in range(E):
        for t in range(1, size_E):
            for i in range(RC.N):
                X[i,:] = x[t+1,i] * x[t,:]
            M = -(M + eta*X)
    
    M = sp.csr_matrix(M)
    RC_heb = ReCom.ESN(N=RC.N, p=RC.p, d=RC.d, rho=RC.rho, sigma=RC.sigma, gamma=RC.gamma, beta=RC.beta)
    RC_heb.M = M
    RC_heb.W_in = RC.W_in.copy()
    training_traj_heb = RC_heb.train(train_data, t_points_training, t_listen)
    return RC_heb


# def NRMSE_mat(a_space, b_space, orbits, val_data):
    
#     a, b = np.meshgrid(a_space, b_space)
#     param_space = np.array([a.flatten(),b.flatten()]).T
#     NRMSE = dict(keys=param_space)
#     norms = np.asarray([np.linalg.norm(val_data[t]) for t in range(val_data.shape[0])])
#     mean_norm = np.mean(norms)
#     square_dist = np.asarray([np.linalg.norm(val_data[t]-mean_norm)**2 for t in range(val_data.shape[0])])
#     sd = np.sqrt(square_dist.sum()/val_data.shape[0])
    
#     for a, b in param_space:
#         x = orbits[(a, b)]
#         NRMSE[(a, b)] = pred_NRMSE(x, val_data, sd)
            
#     bifurcation_matrix = np.zeros((a_space.size, b_space.size))
    
#     for i in range(a_space.size):
#         for j in range(b_space.size):
#             bifurcation_matrix[i, j] = NRMSE[(a_space[i], b_space[j])]
    
#     return bifurcation_matrix