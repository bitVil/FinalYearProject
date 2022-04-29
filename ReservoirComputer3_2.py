# -*- coding: utf-8 -*-
"""

Created on Mon March 21 12:46:32 2022

@author: David Fox
"""

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import splprep, splev
import scipy.integrate as integrate
from scipy import stats

    
    
class ESN(object):
    """
    Implementation of an echo state network (ESN).
    
    Instance Variables:
            seed (int): Random seed, used in generating W_in_orig and M_orig.
            N (int): Dimension of hidden layer.
            d (int): Dimension of input layer.
            p (double): Density of hidden layer connections.
            rho (double): Spectral radius of adjacency matrix M.
            gamma (double): Controls time scale of hidden layer dynamics.
            sigma (double): Scales input signal strength.
            W_in_orig (Nxd sparse csr matrix): Matrix describing connections from input layer to hidden layer.
            W_in (Nxd sparse csr matrix): W_in_orig, scaled by sigma.
            M_orig (NxN sparse csr matrix): Adjacency matrix for hidden layer connections.
            M (NxN sparse csr matrix): M_orig, scaled to have spectral radius rho.
            W_out (dx2N matrix): Output layer connections learned during training.
            beta (double): Regularization parameter used when training W_out.
            r_T (Nd array): Hidden layer state after training.
            
    """
    
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
        self.r_T = None
        self.u = None
        
        
    @property
    def rho(self):
        return self.__rho
    
    
    @rho.setter
    def rho(self, rho):
        """
        Changes adjacency matrix M by rescaling M_orig after resetting rho.

        """
        self.__rho = rho
        self.M = rho * self.M_orig
        
        
    @property
    def sigma(self):
        return self.__sigma

    
    @sigma.setter
    def sigma(self, sigma):
        """
        Changes input matrix W_in by rescaling W_in_orig after resetting sigma.

        """
        self.__sigma = sigma
        self.W_in = sigma * self.W_in_orig
    
    
    def f_PR(self, r, t, *args):
        """
        Vector field giving dynamics of predicting reservoir system.

        Parameters
        ----------
        r : Nd array
            Vector of reservoir state variables
        t : double
            Time variable for reservoir state
        *args : 
            Unused paramater, included in function params for compatability with 
            scipy.integrate.odeint.

        Returns
        -------
        Nd array
            Value of function giving predicting reservoir dynamics for given r 
            and t values.

        """
        v = self.W_out.dot(self.q(r))
        return self.gamma*(-r + np.tanh(self.M.dot(r) + self.W_in.dot(v)))
    
    
    def f_LR(self, r, t, *args):
        """
        Vector field giving dynamics of listening reservoir system.

        Parameters
        ----------
        r : Nd array
            Vector of reservoir state variables
        t : double
            Time variable for reservoir state
        *args : 
            Unused paramater, included in function params for compatability with 
            scipy.integrate.odeint.

        Returns
        -------
        Nd array
            Value of function giving listening reservoir dynamics for given r 
            and t values.

        """
        return self.gamma*(-r + np.tanh(self.M.dot(r) + self.W_in.dot(self.u(t))))
        
        
    def adj_matrix(self):
        """
        Generates a random Erdos-Renyi NxN sparse csr matrix scaled by its max
        eigenvalue.

        Returns
        -------
        NxN sparse csr matrix
            Adjacency matrix M scaled to have unit spectral radius.

        """
        
        np.random.seed(seed = self.seed)
        rvs = stats.uniform(loc=-1, scale=2).rvs
        M = sp.random(self.N, self.N, self.p, format='csr', data_rvs=rvs)
        max_eval = np.abs(sp.linalg.eigs(M, 1, which='LM', return_eigenvectors=False)[0])
        
        return (1/max_eval)*M


    def input_matrix(self):
        """
        Generates a random Nxd sparse csr matrix with 1 entry in each row 
        sampled from UNIF(-1,1).

        Returns
        -------
        Nxd sparse csr matrix
            Random matrix giving connections from input layer to reservoir.

        """
        
        
        np.random.seed(seed = self.seed)
        # Create sparse matrix in COO form, then cast to CSR form.
        rows = np.arange(0, self.N)
        cols = stats.randint(0, self.d).rvs(self.N)
        values = stats.uniform(loc=-1, scale=2).rvs(self.N)
        W_in = sp.coo_matrix((values, (rows, cols))).tocsr()
        
        return W_in
    
    
    def q(self, r):
        """
        Function q() used as part of output function used for predictions.

        Parameters
        ----------
        r : Nd array
            Vector of reservoir state variables.

        Returns
        -------
        x : 2xNd array
            Vector of reservoir state variables, concatenated with squares of
            reservoir state variables.
        """
        x = np.zeros(2*self.N)
        x[0:self.N] = r
        x[self.N: 2*self.N] = r**2
        
        return x
    

    def spline(self, data, t):
        """
        

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        t : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        coords = [data[:,i] for i in range(self.d)]
        tck, u = splprep(coords, u = t, s=0)
        return lambda t: np.asarray(splev(t, tck))
    
    
    def train(self, data, t, t_listen):
        """
        

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        t : TYPE
            DESCRIPTION.
        t_listen : TYPE
            DESCRIPTION.

        Returns
        -------
        LR_traj : TYPE
            DESCRIPTION.

        """
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
    
    
    def predict(self, t_predict, data=None, t=None):
        """
        

        Parameters
        ----------
        t_predict : TYPE
            DESCRIPTION.
        data : TYPE, optional
            DESCRIPTION. The default is None.
        t : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        prediction : TYPE
            DESCRIPTION.

        """
        if type(data) == type(None):
            PR_traj = integrate.odeint(self.f_PR, self.r_T, t_predict)
        else:
            self.u = self.spline(data, t)
            LR_traj = integrate.odeint(self.f_LR, np.zeros(self.N), t)
            PR_traj = integrate.odeint(self.f_PR, LR_traj[-1], t_predict)
            
        prediction = np.asarray([self.W_out.dot(self.q(p)) for p in PR_traj])
        return prediction
        
        
class SPESN(ESN):
    """
    Implementation of an echo state network (ESN) augmented with an anti-Hebbian
    synaptic plasticity rule. Subclass of ESN.
        
    Instance Variables:
            eta_s (double): Learning rate for synaptic plasticity rule.
            epochs_s (int): Number of training epochs for plasticity rule.
    """
          
    def __init__(self, N, p, d, rho, sigma, gamma, beta, eta_s, epochs_s, seed=1):
        ESN.__init__(self, N, p, d, rho, sigma, gamma, beta, seed)
        self.eta_s = eta_s
        self.epochs_s = epochs_s
        
        
    def f_PR(self, r, t, *args):
        v = self.W_out.dot(self.q(r))
        f = self.gamma*(-r + np.tanh(self.M.dot(r) + self.W_in.dot(v)))
        return np.squeeze(np.asarray(f))
    
    
    def f_LR(self, r, t, *args):
        f = self.gamma*(-r + np.tanh(self.M.dot(r) + self.W_in.dot(self.u(t))))
        return np.squeeze(np.asarray(f))
    
    
    def SP_train(self, data, t_points, t_listen, reset_M=True, scale_M=0):
        if reset_M:
            self.rho = self.rho 
        self.u = self.spline(data, t_points)
    
        for e in range(self.epochs_s):
            x = integrate.odeint(self.f_LR, np.zeros(self.N), t_points)
            for t in range(t_listen, data.shape[0]-1):
                self.M = self.M - self.eta_s*(np.asarray([x[t+1]]).T)@np.asarray([x[t]])
                
            # If scale_M==1, scale M after each epoch to have spectral radius rho.
            if scale_M==1:
                self.M = sp.csr_matrix(self.M)
                max_eval = np.abs(sp.linalg.eigs(self.M, 1, which='LM', return_eigenvectors=False)[0])
                self.M = (self.rho/max_eval)*self.M
            
        
        # If scale_M==2, scale M once after training to have spectral radius rho.
        if scale_M==2:
                self.M = sp.csr_matrix(self.M)
                max_eval = np.abs(sp.linalg.eigs(self.M, 1, which='LM', return_eigenvectors=False)[0])
                self.M = (self.rho/max_eval)*self.M
                
        # If scale_M==0, don't rescale M during training.
        else:
                self.M = sp.csr_matrix(self.M)
        
        return None

    
        
class IPESN(SPESN):
    def __init__(self, N, p, d, rho, sigma, gamma, beta, eta_i, epochs_i, eta_s=0, epochs_s=0, mu=0, sd=0.5, seed=1):
        SPESN.__init__(self, N, p, d, rho, sigma, gamma, beta, eta_s, epochs_s, seed)
        self.eta_i = eta_i
        self.epochs_i = epochs_i
        self.mu=mu
        self.sd=sd
        self.a = np.ones(N)
        self.b = np.zeros(N)
    
    
    def f_PR(self, r, t, *args):
        v = self.W_out.dot(self.q(r))
        f = self.gamma*(-r + np.tanh(self.M.dot(r) + self.W_in.dot(v) + self.b))
        return np.squeeze(np.asarray(f))
    
    
    def f_LR(self, r, t, *args):
        f = self.gamma*(-r + np.tanh(self.M.dot(r) + self.W_in.dot(self.u(t)) + self.b))
        return np.squeeze(np.asarray(f))
    
    
    def H(self, x):
        return -self.mu/self.sd**2 + (x/self.sd**2)*(2*self.sd**2 + 1 - x**2 + self.mu*x)
    
    
    def IP_train(self, data, t_points, t_listen, reset_M=True):
        # If reset_M==True, change M, W_in, a and b to original values. 
        if reset_M:
            self.rho = self.rho
            self.sigma = self.sigma
            self.a = np.ones(self.N)
            self.b = np.zeros(self.N)
        self.u = self.spline(data, t_points)

        
        for e in range(self.epochs_i):
            x = integrate.odeint(self.f_LR, np.zeros(self.N), t_points)
            z = np.asarray([self.M.dot(x[t]) + self.W_in.dot(self.u(t_points[t])) for t in range(len(t_points))])
            for t in range(t_listen, data.shape[0]-1):
                delta_b = -self.eta_i*self.H(x[t])
                delta_a = self.eta_i/self.a + np.dot(np.diag(delta_b), z[t])
                self.b += delta_b
                self.a += delta_a
        
        self.M = self.rho*np.matmul(np.diag(self.a), self.M.todense())
        self.W_in = self.sigma*np.matmul(np.diag(self.a), self.W_in.todense())