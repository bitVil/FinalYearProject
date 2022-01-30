# -*- coding: utf-8 -*-
"""

Created on Tue July 06 12:46:32 2021

@author: David Fox
"""

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import splprep, splev
import scipy.integrate as integrate
from scipy import stats

    
    
class ESN(object):
    def __init__(self, N, p, d, rho, sigma, gamma, beta, seed=1):      
        # Create input layer.
        self.N = N
        self.d = d
        self.seed = seed
        self.W_in_orig = self.input_matrix()
        self.sigma = sigma
        
        # Create reservoir adjacency matrix.
        self.p = p
        self.M_orig = self.adj_matrix()
        self.rho = rho
        
        # Output layer initially 'None' before training.
        self.W_out = None
    
        self.gamma = gamma
        self.beta = beta
        
        self.u = None
        self.W_out = None
        self.r_T = None
        
        
    @property
    def rho(self):
        return self.__rho
    
    
    @rho.setter
    def rho(self, rho):
        self.__rho = rho
        self.M = rho * self.M_orig
        
        
    @property
    def sigma(self):
        return self.__sigma

    
    @sigma.setter
    def sigma(self, sigma):
        self.__sigma = sigma
        self.W_in = sigma * self.W_in_orig
    
    
    def f_PR(self, r, t, *args):
        v = self.W_out.dot(self.q(r))
        return self.gamma*(-r + np.tanh(self.M.dot(r) + self.W_in.dot(v)))
    
    
    def f_LR(self, r, t, *args):
        return self.gamma*(-r + np.tanh(self.M.dot(r) + self.W_in.dot(self.u(t))))
        
        
    def adj_matrix(self):
        """
        Generates a random Erdos-Renyi NxN sparse csr matrix and its max eigenvalue.
        """
        
        np.random.seed(seed = self.seed)
        rvs = stats.uniform(loc=-1, scale=2).rvs
        M = sp.random(self.N, self.N, self.p, format='csr', data_rvs=rvs)
        max_eval = np.abs(sp.linalg.eigs(M, 1, which='LM', return_eigenvectors=False)[0])
        
        return (1/max_eval)*M


    def input_matrix(self):
        """
        Generates a random Nxd sparse csr matrix with 1 entry in each 
        row sampled from UNIF(-1,1).
        """
        
        np.random.seed(seed = self.seed)
        # Create sparse matrix in COO form, then cast to CSR form.
        rows = np.arange(0, self.N)
        cols = stats.randint(0, self.d).rvs(self.N)
        values = stats.uniform(loc=-1, scale=2).rvs(self.N)
        W_in = sp.coo_matrix((values, (rows, cols))).tocsr()
        
        return W_in
    
    
    def q(self, r):
        x = np.zeros(2*self.N)
        x[0:self.N] = r
        x[self.N: 2*self.N] = r**2
        
        return x
    

    def spline(self, data, t):
        coords = [data[:,i] for i in range(self.d)]
        tck, u = splprep(coords, u = t, s=0)
        return lambda t: np.asarray(splev(t, tck))
    
    
    def hebb_learn(self, data, t, eta, E, size_E=9999):
        self.u = self.spline(data, t)
        x = integrate.odeint(self.f_LR, np.zeros(self.N), t)
        X = np.zeros((self.N,self.N))
        M = self.M.copy()
    
        for e in range(E):
            for t in range(1, size_E):
                for i in range(self.N):
                    X[i,:] = x[t+1,i] * x[t,:]
                M = -(M + eta*X)
        
        self.M = sp.csr_matrix(M)
        return None
    
    
    def train(self, data, t, t_listen):
        # Integrate Listening reservoir system
        if type(self.u) == type(None):    
            self.u = self.spline(data, t)
        LR_traj = integrate.odeint(self.f_LR, np.zeros(self.N), t)
        
        X = np.zeros((2*self.N, t.size - t_listen))
        Y = np.transpose(data[t_listen:])
        for i in range(t_listen, t.size - 1):
              X[:,i+1 - t_listen] = self.q(LR_traj[i+1])
                    
        # Calculate output matrix.
        X_T = np.transpose(X)
        M_1 = np.matmul(Y, X_T)
        M_2 = np.linalg.inv(np.matmul(X, X_T) + self.beta*np.identity(2*self.N))
        self.W_out = np.matmul(M_1, M_2)
        self.r_T = LR_traj[-1]
        
        return LR_traj
        
    
    def predict(self, t):
        PR_traj = integrate.odeint(self.f_PR, self.r_T, t)
        prediction = np.asarray([self.W_out.dot(self.q(p)) for p in PR_traj])
        return prediction
