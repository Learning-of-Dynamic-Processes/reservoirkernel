
# %% #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 20 13:03:48 2022
Updated on April 27 2024

@author: Lukas Gonon, Lyudmila Grigoryeva, and Juan-Pablo Ortega
We acknowledge that we compare the reservoir kernel with other kernels using 
the codes and the packge sigkernel available at https://github.com/crispitagorico/sigkernel
"""

# %%
import time
import signal
import hickle as hkl
from scipy.stats import uniform, randint, loguniform, wasserstein_distance
from numpy.linalg import lstsq
from tslearn.backend import instantiate_backend
import iisignature
from numba import njit
from sklearn.base import BaseEstimator, RegressorMixin
from functools import partial
from tslearn.metrics import gak, sigma_gak
from sklearn.metrics.pairwise import pairwise_kernels
import plot_funcs as plots
from sklearn.preprocessing import KernelCenterer, MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from tqdm import tqdm
from sklearn.multioutput import MultiOutputRegressor
import torch
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, HalvingGridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer, accuracy_score, mean_squared_error, r2_score, median_absolute_error
import scipy.io as spio
from sklearn.svm import SVR
import sigkernel
import perf_funcs as perf
import mat73
import volt_funcs as volt
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def neg_calculate_nmse(y_true, y_pred):
    # Calculate MSE
    mse = np.mean((y_true - y_pred)**2, axis=0)
    # print(np.shape(y_true))
    # print(np.shape(y_pred))
    # print(np.shape(mse))
    # Calculate MSE
    factor = np.mean((y_true)**2, axis=0)
    neg_nmse = -np.mean(mse / factor)
    
    return neg_nmse

def mean_wasserstein_dist(y_true, y_pred):
    was_dist=[]
    ndim=np.shape(y_true)[1]
    for i in range(ndim):
        was_dist = np.append(was_dist, wasserstein_distance(y_true[:,i], y_pred[:,i]))
    return np.mean(was_dist)


def neg_calculate_mse(y_true, y_pred):
    # Calculate MSE
    neg_mse = -np.mean((y_true - y_pred)**2)
    
    return neg_mse

def set_scorer(scorer='None'):
    if scorer=='neg_mse':
        my_scorer = {'neg_mse': make_scorer(neg_calculate_mse, greater_is_better=True)}  
        my_refit = 'neg_mse' 
        score_str = 'mean_test_neg_mse'
    elif scorer=='neg_nmse':
        my_scorer = {'neg_nmse': make_scorer(neg_calculate_nmse, greater_is_better=True)}  
        my_refit = 'neg_nmse' 
        score_str = 'mean_test_neg_nmse'
    elif scorer=='score':
        my_scorer=None
        my_refit=True
        score_str = 'mean_test_score'
        
    return my_scorer, my_refit, score_str
######################################################
################ GAK Class ###########################
######################################################
def gak_kernel_callable(X, Y, sigma):
    nT_X = np.shape(X)[0]
    nd_X = np.shape(X)[1]
    nT_Y = np.shape(Y)[0]
    nd_Y = np.shape(Y)[1]
    gak_kernel = np.zeros((nT_X,nT_Y))
    if np.array_equal(X,Y):
        for t in range(0, nT_X):
            for j in range(t+1):
                gak_kernel[t,j] = gak(np.array(X[t,:]), Y[j,:], sigma=sigma)
                #gak_kernel[t,j] = _gak_gram(np.array(np.array(X[t,:]).reshape((-1, nd_X))), np.array(Y[j,:]).reshape((-1, nd_Y)), sigma=sigma)
            gak_kernel[j,t] = gak_kernel[t,j]
    else:
        for t in range(0, nT_X):
            for j in range(0, nT_Y):
                gak_kernel[t,j] = gak(np.array(X[t,:]), Y[j,:], sigma=sigma)
                #gak_kernel[t,j] = _gak_gram(np.array(np.array(X[t,:]).reshape((-1, nd_X))), np.array(Y[j,:]).reshape((-1, nd_Y)), sigma=sigma)
            
    return gak_kernel

def _gak_gram(s1, s2, sigma=1.0):
    be = None
    be = instantiate_backend(be, s1)
    #print(s1)
    # s1 = to_time_series(s1, remove_nans=True, be=be)
    # s2 = to_time_series(s2, remove_nans=True, be=be)
    gram = -be.cdist(s1, s2, "sqeuclidean") / (2 * sigma**2)
    gram = be.array(gram)
    gram -= be.log(2 - be.exp(gram))
    print(be.exp(gram))
    return be.exp(gram)

class GAKKernel(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1, sigma=1, ifscale_kernel=False, ifscale_y=False, ifcenter_kernel=False):
        self.sigma = sigma
        self.alpha = alpha
        self.gak_kernel = partial(gak_kernel_callable, sigma=self.sigma)
        self.regressor = KernelRidge(kernel="precomputed", alpha = self.alpha)
        self.ifscale_kernel = ifscale_kernel
        self.ifscale_y = ifscale_y
        self.ifcenter_kernel = ifcenter_kernel

    def fit(self, X, y):
        # Transform to obtain GAK kernel and fit kernel ridge regression
        K_fit = self.gak_kernel(X, X)
        if self.ifcenter_kernel:
            centerer = KernelCenterer()
            K_fit = centerer.fit_transform(K_fit)
            self.centerer = centerer
            
        self.X_fit_ = X.copy()
        # Rescale kernel if needed
        if self.ifscale_kernel:
            self.scaler_K = StandardScaler().fit(K_fit)
            K_fit = self.scaler_K.transform(K_fit)

        if self.ifscale_y:
            self.scaler_y = StandardScaler().fit(y)
            y = self.scaler_y.transform(y)
        
        self.K_fit_ = K_fit
        k = self.regressor
        print(k)
        self.regressor.fit(self.K_fit_, y)
        return self
    
    def predict(self, X):
        # Transform to obtain GAK kernel and predict
        K_test = self.gak_kernel(X, self.X_fit_) # Use X_fit, not K_fit
        if self.ifcenter_kernel:
            centerer = self.centerer
            K_test = centerer.transform(K_test)
            
        if self.ifscale_kernel:
            K_test = self.scaler_K.transform(K_test)

        if self.ifscale_y:
            return self.scaler_y.inverse_transform(self.regressor.predict(K_test))
            
        return self.regressor.predict(K_test)
    
    # Make alpha parameter 'cv-able'
    def set_params(self, **params):
        for k, v in params.items():
            if k == "sigma":
                self.sigma = v
                self.gak_kernel = partial(gak_kernel_callable, sigma=self.sigma)    
            elif k == "alpha":
                self.alpha=v
                self.regressor.alpha = self.alpha    
            else:
                self.regressor = self.regressor.set_params(**{k:v})
                setattr(self.regressor, k, v)
        self.regressor = KernelRidge(kernel="precomputed", alpha = self.alpha)
        print(self.regressor)
        
        return self

######################################################
################ Volterra Class ######################
######################################################

@njit
def _volt_gram_fast_njit(X, Y, omega, tau):
    nT_X, nT_Y = X.shape[0], Y.shape[0]
    # Compute once only instead of in the loop
    omega, tau = omega ** 2, tau ** 2
    # Initialize the gram matrix with ones
    Gram = np.ones((nT_X, nT_Y))
    tau_XY = 1 - X @ Y.T * tau
    # Compute the first row and column of the Gram matrix
    Gram[0, 0] += omega / (1 - omega) / tau_XY[0, 0]
    for i in range(1, nT_X):
        Gram[i, 0] += omega / (1 - omega) / tau_XY[i, 0]
    for i in range(1, nT_Y):
        Gram[0, i] += omega / (1 - omega) / tau_XY[0, i]

    # Compute the rest of the Gram matrix
    for i in range(1, nT_X):
        tau_XY_i = tau_XY[i]
        for j in range(1, nT_Y):
            Gram[i, j] += omega * Gram[i-1, j-1] / tau_XY_i[j]
    return Gram

def _volt_gram(X, Y, omega, tau):
    nT_X = np.shape(X)[0]
    nT_Y = np.shape(Y)[0]
    Gram = np.zeros((nT_X, nT_Y))
    Gram0 = 1/(1-omega**2)
    for i in range(nT_X):
        for j in range(nT_Y):
            if i==0 or j==0:
                Gram[i, j] = 1 + omega**2 * Gram0/(1-(tau**2)*(np.dot(X[i,:], Y[j,:])))
            else:
                Gram[i, j] = 1 + omega**2 * Gram[i-1,j-1]/(1-(tau**2)*(np.dot(X[i,:], Y[j,:])))
    return Gram
def Volterra_kernel_callable(X, Y, omega, tau, nwashout):
    # print(tau)
    # print(omega)
    volt_K = _volt_gram_fast_njit(X, Y, omega, tau)
            
    return volt_K

class VolterraKernel(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1, omega=0.1, tau=0.1, nwashout=0, ifscale_kernel=False, ifscale_y=False, ifcenter_kernel=False):
        self.alpha = alpha
        self.omega = omega
        self.tau = tau
        self.nwashout = nwashout
        self.Volterra_kernel = partial(Volterra_kernel_callable, omega=self.omega, tau=self.tau, nwashout=self.nwashout)
        self.regressor = KernelRidge(kernel="precomputed", alpha=self.alpha)
        self.ifscale_kernel = ifscale_kernel
        self.ifscale_y = ifscale_y
        self.ifcenter_kernel = ifcenter_kernel

    def fit(self, X, y):
        # Transform to obtain Volterra kernel and fit kernel ridge regression
        K_fit = self.Volterra_kernel(X, X)
        K_fit = K_fit[self.nwashout:, self.nwashout:]
        if self.ifcenter_kernel:
            centerer = KernelCenterer()
            K_fit = centerer.fit_transform(K_fit)
            self.centerer = centerer
            
        self.X_fit_ = X.copy()
        # Rescale kernel if needed
        if self.ifscale_kernel:
            self.scaler_K = StandardScaler().fit(K_fit)
            K_fit = self.scaler_K.transform(K_fit)

        if self.ifscale_y:
            self.scaler_y = StandardScaler().fit(y)
            y = self.scaler_y.transform(y)

        self.K_fit_ = K_fit
        #print(self.Volterra_kernel)
        self.regressor.fit(self.K_fit_, y[self.nwashout:,:])
        return self
    
    def predict(self, X):
        # Transform to obtain Volterra kernel and predict
        K_test = self.Volterra_kernel(X, self.X_fit_) # Use X_fit, not K_fit
        K_test = K_test[:, self.nwashout:]
        if self.ifcenter_kernel:
            centerer = self.centerer
            K_test = centerer.transform(K_test)
            
        if self.ifscale_kernel:
            K_test = self.scaler_K.transform(K_test)

        if self.ifscale_y:
            return self.scaler_y.inverse_transform(self.regressor.predict(K_test))
            
        return self.regressor.predict(K_test)
    
    # Make alpha parameter 'cv-able'
    def set_params(self, **params):
        for k, v in params.items():
            if k == "nwashout":
                self.nwashout = v
            if k == "omega":
                self.omega = v
            elif k == "tau":
                self.tau = v
            elif k == "alpha":
                self.alpha=v
                self.regressor.alpha = self.alpha    
            else:
                setattr(self.regressor, k, v)
        if self.omega > np.sqrt(1 - (self.tau**2)):
            print(self.tau)
            print(self.omega)
            self.omega = np.sqrt(1 - (self.tau**2))*0.99
            print('became:')
            print(self.omega)
        self.Volterra_kernel = partial(Volterra_kernel_callable, 
                                       omega=self.omega, 
                                       tau=self.tau, 
                                       nwashout=self.nwashout)
        self.regressor = KernelRidge(kernel="precomputed", alpha = self.alpha)
        #print(self.Volterra_kernel)
        
        return self



######################################################
################ SigPDEKernel Class ##################
######################################################
def sigPDE_kernel_callable(X, Y, sigma):
    # Specify the static kernel 
    static_kernel = sigkernel.RBFKernel(sigma=sigma)

    # Initialize the corresponding signature kernel
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=0)
    
    X = torch.tensor(X, dtype=torch.float64, device='cpu')
    Y = torch.tensor(Y, dtype=torch.float64, device='cpu')
    
    # Gram matrix
    if np.array_equal(X,Y):
        K = signature_kernel.compute_Gram(X, Y, sym=True).numpy()
    else:
        K = signature_kernel.compute_Gram(X, Y, sym=False).numpy()
    
    return K
class SigPDEKernel(BaseEstimator, RegressorMixin):
    def __init__(self, alpha = 1, sigma=1, prescale=1, ifscale_kernel=False, ifscale_y=False):
        self.alpha = alpha
        self.sigma = sigma
        self.sigPDE_kernel = partial(sigPDE_kernel_callable, sigma=self.sigma)
        self.regressor = KernelRidge(kernel="precomputed", alpha = self.alpha)
        self.ifscale_kernel = ifscale_kernel
        self.prescale = prescale
        self.ifscale_y = ifscale_y

    def fit(self, X, y):
        # Transform to obtain kernel and fit kernel ridge regression
        self.max_X = np.max(X)
        X = sigkernel.transform(X/self.max_X, at=True, ll=True, scale=self.prescale)
        K_fit = self.sigPDE_kernel(X, X)
        self.X_fit_ = X.copy()
        # Rescale kernel if needed
        if self.ifscale_kernel:
            self.scaler_K = StandardScaler().fit(K_fit)
            K_fit = self.scaler_K.transform(K_fit)

        if self.ifscale_y:
            self.scaler_y = StandardScaler().fit(y)
            y = self.scaler_y.transform(y)
        
        self.K_fit_ = K_fit
        #print(self.regressor)
        self.regressor.fit(self.K_fit_, y)
        return self
    
    def predict(self, X):
        # Transform to obtain kernel and predict
        X = sigkernel.transform(X/self.max_X, at=True, ll=True, scale=self.prescale)
        K_test = self.sigPDE_kernel(X, self.X_fit_) # Use X_fit, not K_fit
        if self.ifscale_kernel:
            K_test = self.scaler_K.transform(K_test)

        if self.ifscale_y:
            return self.scaler_y.inverse_transform(self.regressor.predict(K_test))
            
        return self.regressor.predict(K_test)
    
    # Make alpha parameter 'cv-able'
    def set_params(self, **params):
        for k, v in params.items():
            if k == "sigma":
                self.sigma = v
                self.sigPDE_kernel = partial(sigPDE_kernel_callable, sigma=self.sigma)   
            elif k == "prescale":
                self.prescale = v    
            elif k == "alpha":
                self.alpha=v
                self.regressor.alpha = self.alpha     
            else:
                setattr(self.regressor, k, v)
        return self
    
######################################################
################ SigTruncKernel Class ################
######################################################
def sigTrunc_cov_callable(X, depth, scale, normalize):
    sig_X = iisignature.sig(scale*X, depth)
    
    # normalization
    if normalize:
        sig_X = sigkernel.normalize(sig_X, X.shape[-1], depth)
    return sig_X
class SigTruncKernel(BaseEstimator, RegressorMixin):
    def __init__(self, kernel="rbf", alpha = 1, gamma = 1, depth = 2, scale = 1, prescale = 1, normalize=False, ifscale_y=False):
        self.kernel = kernel
        self.alpha = alpha
        self.gamma = gamma
        self.prescale = prescale
        self.regressor = KernelRidge(kernel=self.kernel, alpha = self.alpha, gamma = self.gamma)
        self.depth = depth
        self.normalize = normalize
        self.scale = scale
        self.sigTrunc_cov = partial(sigTrunc_cov_callable, depth=self.depth, scale = self.scale, normalize=self.normalize)
        self.ifscale_y = ifscale_y

    def fit(self, X, y):
        self.max_X = np.max(X)
        X = sigkernel.transform(X/self.max_X, at=True, ll=True, scale=self.prescale)
        sig_X = self.sigTrunc_cov(X, depth=self.depth, scale = self.scale, normalize=self.normalize)
        self.X_fit_ = sig_X.copy()

        if self.ifscale_y:
            self.scaler_y = StandardScaler().fit(y)
            y = self.scaler_y.transform(y)
        #print(self.regressor)
        self.regressor.fit(self.X_fit_, y)
        return self
    
    def predict(self, X):
        # Transform to obtain GAK kernel and predict
        X = sigkernel.transform(X/self.max_X, at=True, ll=True, scale=self.prescale)
        sig_test = iisignature.sig(self.scale*X, self.depth)
        
        if self.ifscale_y:
            return self.scaler_y.inverse_transform(self.regressor.predict(sig_test))
            
        return self.regressor.predict(sig_test)
    
    # Make alpha parameter 'cv-able'
    def set_params(self, **params):
        for k, v in params.items():
            if k == "kernel":
                self.kernel = v 
                self.regressor.kernel = self.kernel              
            elif k == "depth":
                self.depth = v        
            elif k == "scale":
                self.scale = v  
            elif k == "prescale":
                self.prescale = v    
            elif k == "alpha":
                self.alpha=v
                self.regressor.alpha = self.alpha 
            elif k == "gamma":
                self.gamma=v
                self.regressor.gamma = self.gamma 
            elif k == "normalize":
                self.normalize = v
                self.sigTrunc_cov = partial(sigTrunc_cov_callable, depth=self.depth, scale = self.scale, normalize=self.normalize)  
            else:
                setattr(self.regressor, k, v)
        self.regressor=KernelRidge(kernel=self.kernel, alpha = self.alpha, gamma = self.gamma)
        #print(self.regressor)
        # print(self.alpha)
        # print(self.gamma)
        return self    


# %%
######################################################
################ Load data ###########################
######################################################
# load data
matstruct_contents = mat73.loadmat("BEKK_d15_data.mat")
returns = matstruct_contents['data_sim']
epsilons = matstruct_contents['exact_epsilons']
Ht_sim_vech = matstruct_contents['Ht_sim_vech']

data_in = epsilons
#data_out = returns
data_out = Ht_sim_vech

ndim = data_in.shape[1]
nT = data_in.shape[0]
ndim_output = data_out.shape[1]

if nT>3760:
    nT = 3760

x = data_in[0:nT-1,:]
#y = data_out[0:nT-1,:]**2*1000
y = data_out[1:nT,0:ndim_output]*1000
t_dispay = 300

# train test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.2, 
    shuffle=False, 
    random_state=1)

scaler_y = StandardScaler().fit(y_train)
y_train_demeaned = scaler_y.transform(y_train)

train_len = np.shape(x_train)[0]
test_len = np.shape(x_test)[0]
total_len = test_len + train_len

M = np.max([np.linalg.norm(z) for z in x])
x_train /= M
x_test /= M

set_my_scorer="score"
my_scorer, my_refit, score_str = set_scorer(set_my_scorer)
     
n_jobs = 100

ifsave=True
ifRandomSearch=True

# %%
################ Linear kernel #######################
######################################################
# Linear kernel
n_iter = 70000
est_string = 'linear'
kr = KernelRidge(kernel="linear")
pipe_kr_linear = Pipeline([('scl', StandardScaler()),
        ('kr_est', kr)])

if ifRandomSearch:
    grid_param_kr_linear = dict(kr_est__alpha=loguniform(a=1e-2, b=1e7))
    krCV_linear = RandomizedSearchCV(
        estimator=pipe_kr_linear,
        param_distributions=grid_param_kr_linear, 
        cv=TimeSeriesSplit(n_splits=5), 
        n_jobs=n_jobs,
        scoring=my_scorer,
        refit=my_refit,
        random_state=0,
        n_iter = n_iter
    )
else:
    grid_param_kr_linear = {
        'kr_est__alpha': np.append(0, np.logspace(-2, 7, n_iter))
    }
    krCV_linear = GridSearchCV(
        estimator=pipe_kr_linear,
        param_grid=grid_param_kr_linear, 
        cv=TimeSeriesSplit(n_splits=5), 
        n_jobs=n_jobs,
        scoring=my_scorer,
        refit=my_refit
    )
# Start the timer
start_time = time.time()
krCV_linear.fit(x_train, y_train_demeaned)
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time

print("RandomizedSearchCV took {:.2f} seconds.".format(elapsed_time))
score = krCV_linear.cv_results_[score_str]
#print(pd.DataFrame(krCV_linear.cv_results_).sort_values(by=score_str))
pred_y_test_linear=krCV_linear.predict(x_test)
pred_y_test_linear = scaler_y.inverse_transform(pred_y_test_linear)

nmsekr_linear = perf.calculate_nmse(y_test, pred_y_test_linear)
# Print testing normalized mean squared error
print("Testing NMSE: ", nmsekr_linear)
plt.plot(pred_y_test_linear[0:t_dispay,0])
plt.plot(y_test[0:t_dispay,0])

time_string = 'mean_'+ set_my_scorer+'_time'
mean_fit_time= krCV_linear.cv_results_['mean_fit_time']
mean_est_time= krCV_linear.cv_results_[time_string]
n_splits  = krCV_linear.n_splits_ #number of splits of training data
n_iter = pd.DataFrame(krCV_linear.cv_results_).shape[0] #Iterations per split

##print(np.mean(mean_fit_time + mean_est_time) * n_splits * n_iter)

if ifsave:
    # Saving Variables
    file_path = 'data_score_' + str(ndim)+'d_'+set_my_scorer+ '_'+ est_string+ '_'+ '.pickle'

    # write variables to filename
    hkl.dump([ifRandomSearch, elapsed_time, krCV_linear.best_estimator_, krCV_linear.best_params_,
                     krCV_linear.best_score_,
                    nmsekr_linear, pred_y_test_linear, y_test], file_path)

    print("The file has been saved successfully.")

# with open(file_path, 'rb') as f:
#     krCV_linear.best_estimator_, krCV_linear.best_params_,
#                      krCV_linear.best_score_,
#                   nmsekr_linear = pickle.load(f)
print(krCV_linear.best_params_)
print(krCV_linear.best_score_)
print('Linear done')
# %%
################ RBF kernel ##########################
######################################################
# rbf kernel
est_string = 'rbf'
kr = KernelRidge(kernel="rbf")
pipe_kr_rbf = Pipeline([('scl', StandardScaler()),
        ('kr_est', kr)])
#pipe_kr_rbf = Pipeline([('kr_est', kr)])
n_iter = 50000
loc_log_alpha = -9
scale_log_alpha = -2
loc_log_gamma = -6
scale_log_gamma = -3
num_params = 2

if ifRandomSearch:
    grid_param_kr_rbf = dict(kr_est__alpha=loguniform(a=10**loc_log_alpha, b=10**scale_log_alpha),
                             kr_est__gamma=loguniform(a=10**loc_log_gamma, b=10**scale_log_gamma))
    krCV_rbf = RandomizedSearchCV(
        estimator=pipe_kr_rbf,
        param_distributions=grid_param_kr_rbf, 
        cv=TimeSeriesSplit(n_splits=5), 
        n_jobs=n_jobs,
        scoring=my_scorer,
        refit=my_refit,
        random_state=0,
        n_iter = n_iter
    )
else:
    grid_param_kr_rbf = {
        'kr_est__alpha': np.append(0, np.logspace(loc_log_alpha, scale_log_alpha, int(n_iter**(num_params**-1))-1)),
        'kr_est__gamma': np.append(1, np.logspace(loc_log_gamma, scale_log_gamma, int(n_iter**(num_params**-1))-1))
    }
    krCV_rbf = GridSearchCV(
        estimator=pipe_kr_rbf,
        param_grid=grid_param_kr_rbf, 
        cv=TimeSeriesSplit(n_splits=5), 
        n_jobs=n_jobs,
        scoring=my_scorer,
        refit=my_refit
    )
# Start the timer
start_time = time.time()
krCV_rbf.fit(x_train, y_train_demeaned)
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print("RandomizedSearchCV took {:.2f} seconds.".format(elapsed_time))

score = krCV_rbf.cv_results_[score_str]
print(pd.DataFrame(krCV_rbf.cv_results_).sort_values(by=score_str))
pred_y_test_rbf=krCV_rbf.predict(x_test)
pred_y_test_rbf = scaler_y.inverse_transform(pred_y_test_rbf)

nmsekr_rbf = perf.calculate_nmse(y_test, pred_y_test_rbf)
# Print testing normalized mean squared error
print("Testing NMSE: ", nmsekr_rbf)
plt.plot(pred_y_test_rbf[0:t_dispay,0])
plt.plot(y_test[0:t_dispay,0])

time_string = 'mean_'+ set_my_scorer+'_time'
mean_fit_time= krCV_rbf.cv_results_['mean_fit_time']
mean_est_time= krCV_rbf.cv_results_[time_string]
n_splits  = krCV_rbf.n_splits_ #number of splits of training data
n_iter = pd.DataFrame(krCV_rbf.cv_results_).shape[0] #Iterations per split

print(np.mean(mean_fit_time + mean_est_time) * n_splits * n_iter)

if ifsave:
    # Saving Variables
    file_path = 'data_score_' + str(ndim)+'d_'+set_my_scorer+ '_'+ est_string+ '_'+ '.pickle'

    # write variables to filename
    hkl.dump([ifRandomSearch, elapsed_time, krCV_rbf.best_estimator_, krCV_rbf.best_params_,
                     krCV_rbf.best_score_,
                    nmsekr_rbf, pred_y_test_rbf, y_test], file_path)
    print("The variable 'data' has been saved successfully.")
    
print(krCV_rbf.best_params_)
print(krCV_rbf.best_score_)
print('rbf done')
# %%
############### Sigmoid kernel ######################
#####################################################
# sigmoid kernel
est_string = 'sigmoid'
kr = KernelRidge(kernel="sigmoid")
pipe_kr_sigmoid = Pipeline([('scl', StandardScaler()),
        ('kr_est', kr)])


n_iter = 4500

loc_log_alpha = -6
scale_log_alpha = -2
loc_log_gamma = -5
scale_log_gamma = -1
loc_log_coef0 = -3
scale_log_coef0 = 0.5
num_params = 3

if ifRandomSearch:
    grid_param_kr_sigmoid = dict(kr_est__alpha=loguniform(a=10**loc_log_alpha, b=10**scale_log_alpha),
                             kr_est__gamma=loguniform(a=10**loc_log_gamma, b=10**scale_log_gamma),
                             kr_est__coef0=loguniform(a=10**loc_log_coef0, b=10**scale_log_coef0))
    krCV_sigmoid = RandomizedSearchCV(
        estimator=pipe_kr_sigmoid,
        param_distributions=grid_param_kr_sigmoid, 
        cv=TimeSeriesSplit(n_splits=5), 
        n_jobs=n_jobs,
        scoring=my_scorer,
        refit=my_refit,
        random_state=0,
        n_iter = n_iter
    )
else:
    grid_param_kr_sigmoid = {
        'kr_est__alpha': np.append(0, np.logspace(loc_log_alpha, scale_log_alpha, int(n_iter**(num_params**-1))-1)),
        'kr_est__gamma': np.append(1, np.logspace(loc_log_gamma, scale_log_gamma, int(n_iter**(num_params**-1))-1)),
        'kr_est__coef0': np.append(0, np.logspace(loc_log_coef0, scale_log_coef0, int(n_iter**(num_params**-1))-1))
    }
    krCV_sigmoid = GridSearchCV(
        estimator=pipe_kr_sigmoid,
        param_grid=grid_param_kr_sigmoid, 
        cv=TimeSeriesSplit(n_splits=5), 
        n_jobs=n_jobs,
        scoring=my_scorer,
        refit=my_refit
    )
# Start the timer
start_time = time.time()
krCV_sigmoid.fit(x_train, y_train_demeaned)
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print("RandomizedSearchCV took {:.2f} seconds.".format(elapsed_time))

score = krCV_sigmoid.cv_results_[score_str]
print(pd.DataFrame(krCV_sigmoid.cv_results_).sort_values(by=score_str))
pred_y_test_sigmoid=krCV_sigmoid.predict(x_test)
pred_y_test_sigmoid = scaler_y.inverse_transform(pred_y_test_sigmoid)

nmsekr_sigmoid = perf.calculate_nmse(y_test, pred_y_test_sigmoid)
# Print testing normalized mean squared error
print("Testing NMSE: ", nmsekr_sigmoid)
plt.plot(pred_y_test_sigmoid[0:t_dispay,0])
plt.plot(y_test[0:t_dispay,0])

time_string = 'mean_'+ set_my_scorer+'_time'
mean_fit_time= krCV_sigmoid.cv_results_['mean_fit_time']
mean_est_time= krCV_sigmoid.cv_results_[time_string]
n_splits  = krCV_sigmoid.n_splits_ #number of splits of training data
n_iter = pd.DataFrame(krCV_sigmoid.cv_results_).shape[0] #Iterations per split

print(np.mean(mean_fit_time + mean_est_time) * n_splits * n_iter)

if ifsave:
    # Saving Variables
    file_path = 'data_score_' + str(ndim)+'d_'+set_my_scorer+ '_'+ est_string+ '_'+ '.pickle'

    # write variables to filename
    hkl.dump([ifRandomSearch, elapsed_time, krCV_sigmoid.best_estimator_, krCV_sigmoid.best_params_,
                     krCV_sigmoid.best_score_,
                    nmsekr_sigmoid, pred_y_test_sigmoid, y_test], file_path)
    print("The variable 'data' has been saved successfully.")
    
print(krCV_sigmoid.best_params_)
print(krCV_sigmoid.best_score_)

print('sigmoid done')
# %%
############### Polynomial kernel ###################
#####################################################
# polynomial kernel
est_string = 'poly'
kr = KernelRidge(kernel="poly")
pipe_kr_poly = Pipeline([('scl', StandardScaler()),
        ('kr_est', kr)])

loc_log_alpha = -3
scale_log_alpha = 0
loc_log_gamma = -4
scale_log_gamma = 0
loc_log_coef0 = -4
scale_log_coef0 = 0
low_degree = 2
high_degree = 4
num_params = 3
num_degree = 2
n_iter = 56000

if ifRandomSearch:
    grid_param_kr_poly = dict(kr_est__alpha=loguniform(a=10**loc_log_alpha, b=10**scale_log_alpha),
                             kr_est__gamma=loguniform(a=10**loc_log_gamma, b=10**scale_log_gamma),
                             kr_est__coef0=loguniform(a=10**loc_log_coef0, b=10**scale_log_coef0),
                             kr_est__degree=[2,3])#randint(low=low_degree, high=high_degree))
    krCV_poly = RandomizedSearchCV(
        estimator=pipe_kr_poly,
        param_distributions=grid_param_kr_poly, 
        cv=TimeSeriesSplit(n_splits=5), 
        n_jobs=n_jobs,
        scoring=my_scorer,
        refit=my_refit,
        random_state=0,
        n_iter = n_iter
    )
else:
    grid_param_kr_poly = {
        'kr_est__alpha': np.append(0, np.logspace(loc_log_alpha, scale_log_alpha, int((n_iter/num_degree)**(num_params**-1))-1)),
        'kr_est__gamma': np.append(1, np.logspace(loc_log_gamma, scale_log_gamma, int((n_iter/num_degree)**(num_params**-1))-1)),
        'kr_est__coef0': np.append(0, np.logspace(loc_log_coef0, scale_log_coef0, int((n_iter/num_degree)**(num_params**-1))-1)),
        'kr_est__degree': [2,3]
    }
    krCV_poly = GridSearchCV(
        estimator=pipe_kr_poly,
        param_grid=grid_param_kr_poly, 
        cv=TimeSeriesSplit(n_splits=5), 
        n_jobs=n_jobs,
        scoring=my_scorer,
        refit=my_refit
    )

# Start the timer
start_time = time.time()
krCV_poly.fit(x_train, y_train_demeaned)
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print("RandomizedSearchCV took {:.2f} seconds.".format(elapsed_time))

score = krCV_poly.cv_results_[score_str]
#print(pd.DataFrame(krCV_poly.cv_results_).sort_values(by=score_str))
pred_y_test_poly=krCV_poly.predict(x_test)
pred_y_test_poly = scaler_y.inverse_transform(pred_y_test_poly)

nmsekr_poly = perf.calculate_nmse(y_test, pred_y_test_poly)
# Print testing normalized mean squared error
print("Testing NMSE: ", nmsekr_poly)
plt.plot(pred_y_test_poly[0:t_dispay,2])
plt.plot(y_test[0:t_dispay,2])

time_string = 'mean_'+ set_my_scorer+'_time'
mean_fit_time= krCV_poly.cv_results_['mean_fit_time']
mean_est_time= krCV_poly.cv_results_[time_string]
n_splits  = krCV_poly.n_splits_ #number of splits of training data
n_iter = pd.DataFrame(krCV_poly.cv_results_).shape[0] #Iterations per split

print(np.mean(mean_fit_time + mean_est_time) * n_splits * n_iter)

if ifsave:
    # Saving Variables
    file_path = 'data_score_' + str(ndim)+'d_'+set_my_scorer+ '_'+ est_string+ '_'+ '.pickle'

    # write variables to filename
    hkl.dump([ifRandomSearch, elapsed_time, krCV_poly.best_estimator_, krCV_poly.best_params_,
                     krCV_poly.best_score_,
                    nmsekr_poly, pred_y_test_poly, y_test], file_path)
    print("The variable 'data' has been saved successfully.")

print(krCV_poly.best_params_)
print(krCV_poly.best_score_)
print('poly done')
# %%


################ PDE SIGNATURE kernel ################
######################################################
# PDE signature kernel
est_string = 'PDEsign'
kr = SigPDEKernel(ifscale_y=False)
pipe_kr_PDEsign = Pipeline(
    #[('scl', StandardScaler()),
    [('kr_est', kr)]
)

loc_log_alpha = -5
scale_log_alpha = 5
loc_log_sigma = -5
scale_log_sigma = 0
loc_log_prescale = -2
scale_log_prescale = 0
num_params = 3

n_iter = 100

if ifRandomSearch:
    grid_param_PDEsign = dict(kr_est__alpha=loguniform(a=10**loc_log_alpha, b=10**scale_log_alpha),
                             kr_est__sigma=loguniform(a=10**loc_log_sigma, b=10**scale_log_sigma),
                             kr_est__prescale=loguniform(a=10**loc_log_prescale, b=10**scale_log_prescale))
    krCV_PDEsign = RandomizedSearchCV(
        estimator=pipe_kr_PDEsign,
        param_distributions=grid_param_PDEsign, 
        cv=TimeSeriesSplit(n_splits=5), 
        n_jobs=n_jobs,
        scoring=my_scorer,
        refit=my_refit,
        random_state=0,
        n_iter = n_iter
    )
else:
    grid_param_PDEsign = {
        'kr_est__alpha': np.append(0, np.logspace(loc_log_alpha, scale_log_alpha, int((n_iter/num_degree)**(num_params**-1))-1)),
        'kr_est__sigma': np.append(1, np.logspace(loc_log_sigma, scale_log_sigma, int((n_iter/num_degree)**(num_params**-1))-1)),
        'kr_est__prescale': np.append(1, np.logspace(loc_log_prescale, scale_log_prescale, int((n_iter/num_degree)**(num_params**-1))-1))
    }
    krCV_PDEsign = GridSearchCV(
        estimator=pipe_kr_PDEsign,
        param_grid=grid_param_PDEsign, 
        cv=TimeSeriesSplit(n_splits=5), 
        n_jobs=n_jobs,
        scoring=my_scorer,
        refit=my_refit
    )

# Start the timer
start_time = time.time()
krCV_PDEsign.fit(x_train, y_train_demeaned)
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print("RandomizedSearchCV took {:.2f} seconds.".format(elapsed_time))


score = krCV_PDEsign.cv_results_[score_str]
#print(pd.DataFrame(krCV_PDEsign.cv_results_).sort_values(by=score_str))
pred_y_test_PDEsign=krCV_PDEsign.predict(x_test)
pred_y_test_PDEsign = scaler_y.inverse_transform(pred_y_test_PDEsign)

nmsekr_PDEsign = perf.calculate_nmse(y_test, pred_y_test_PDEsign)
# Print testing normalized mean squared error
print("Testing NMSE: ", nmsekr_PDEsign)
plt.plot(pred_y_test_PDEsign[0:t_dispay,1])
plt.plot(y_test[0:t_dispay,1])

time_string = 'mean_'+ set_my_scorer+'_time'
mean_fit_time= krCV_PDEsign.cv_results_['mean_fit_time']
mean_est_time= krCV_PDEsign.cv_results_[time_string]
n_splits  = krCV_PDEsign.n_splits_ #number of splits of training data
n_iter = pd.DataFrame(krCV_PDEsign.cv_results_).shape[0] #Iterations per split

print(np.mean(mean_fit_time + mean_est_time) * n_splits * n_iter)

if ifsave:
    # Saving Variables
    file_path = 'data_score_' + str(ndim)+'d_'+set_my_scorer+ '_'+ est_string+ '_'+ '.pickle'

    # write variables to filename
    hkl.dump([krCV_PDEsign.best_estimator_, krCV_PDEsign.best_params_,
                     krCV_PDEsign.best_score_,
                    nmsekr_PDEsign, elapsed_time, pred_y_test_PDEsign, y_test], file_path)
    print("The variable 'data' has been saved successfully.")

print(krCV_PDEsign.best_params_)
print('PDEsign done')
# %%
################ VOLTERRA kernel ################
######################################################
# Volterra kernel
est_string = 'Volterra'

loc_log_alpha = -5
scale_log_alpha = 1
loc_log_omega = -2
scale_log_omega = 0
loc_log_tau = -1
scale_log_tau = -0.002
num_params = 3
n_iter = 500
num_tau = 100
tau_set = np.logspace(loc_log_tau, scale_log_tau, num_tau)
best_score_value=-1e8
# Start the timer
start_time = time.time()
if ifRandomSearch:
    for tau in tau_set:
        kr = VolterraKernel(ifscale_y=False, tau=tau)
        pipe_kr_Volterra = Pipeline(
            #[('scl', StandardScaler()),
            [('kr_est', kr)]
        )
        #print(np.sqrt(1 - (tau**2))*0.99)
        grid_param_Volterra = dict(kr_est__alpha=loguniform(a=10**loc_log_alpha, b=10**scale_log_alpha),
                                kr_est__omega=loguniform(a=10**loc_log_omega, b=np.sqrt(1 - (tau**2))*0.999))
        krCV_Volterra = RandomizedSearchCV(
            estimator=pipe_kr_Volterra,
            param_distributions=grid_param_Volterra, 
            cv=TimeSeriesSplit(n_splits=5), 
            n_jobs=n_jobs,
            scoring=my_scorer,
            refit=my_refit,
            random_state=0,
            n_iter = n_iter
        )
        krCV_Volterra.fit(x_train, y_train_demeaned)
        if krCV_Volterra.best_score_>best_score_value:
            print(krCV_Volterra.best_score_)
            # print(krCV_Volterra.best_estimator_)
            # print(tau)
            #print(pd.DataFrame(krCV_Volterra.cv_results_).sort_values(by=score_str))
            best_score_value = krCV_Volterra.best_score_
            best_krCV_Volterra= krCV_Volterra
            best_tau = tau
else:
    for tau in tau_set:
        kr = VolterraKernel(ifscale_y=False, tau=tau)
        pipe_kr_Volterra = Pipeline(
            #[('scl', StandardScaler()),
            [('kr_est', kr)]
        )
        print(np.sqrt(1 - (tau**2))*0.99)
        grid_param_Volterra = {
            'kr_est__alpha': np.append(0, np.logspace(loc_log_alpha, scale_log_alpha, int((n_iter)**(num_params**-1))-1)),
            'kr_est__omega': np.logspace( loc_log_omega, np.log10(np.sqrt(1 - (tau**2))*0.99), int((n_iter)**(num_params**-1)))
        }
        krCV_Volterra = GridSearchCV(
            estimator=pipe_kr_Volterra,
            param_grid=grid_param_Volterra, 
            cv=TimeSeriesSplit(n_splits=5), 
            n_jobs=n_jobs,
            scoring=my_scorer,
            refit=my_refit
        )
        krCV_Volterra.fit(x_train, y_train_demeaned)
        print(krCV_Volterra.best_score_)
        if krCV_Volterra.best_score_>best_score_value:
            print(krCV_Volterra.best_score_)
            print(krCV_Volterra.best_estimator_)
            print(tau)
            #print(pd.DataFrame(krCV_Volterra.cv_results_).sort_values(by=score_str))
            best_score_value = krCV_Volterra.best_score_
            best_krCV_Volterra= krCV_Volterra
            best_tau = tau
    
    
krCV_Volterra=best_krCV_Volterra       
print(krCV_Volterra.best_estimator_)       
print(best_tau)                      

#krCV_Volterra.fit(x_train, y_train_demeaned)

# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = (end_time - start_time)/len(tau_set)
print("SearchCV took {:.2f} seconds.".format(elapsed_time))

score = krCV_Volterra.cv_results_[score_str]
print(pd.DataFrame(krCV_Volterra.cv_results_).sort_values(by=score_str))
pred_y_test_Volterra=krCV_Volterra.predict(x_test)
pred_y_test_Volterra = scaler_y.inverse_transform(pred_y_test_Volterra)

nmsekr_Volterra = perf.calculate_nmse(y_test, pred_y_test_Volterra)
# Print testing normalized mean squared error
print("Testing NMSE: ", nmsekr_Volterra)
plt.plot(pred_y_test_Volterra[0:t_dispay,0])
plt.plot(y_test[0:t_dispay,0])

if ifsave:
    # Saving Variables
    file_path = 'data_score_' + str(ndim)+'d_'+set_my_scorer+ '_'+ est_string+ '_'+ '.pickle'

    # write variables to filename
    hkl.dump([ifRandomSearch, krCV_Volterra.best_estimator_, krCV_Volterra.best_params_,
                     krCV_Volterra.best_score_,
                    nmsekr_Volterra, elapsed_time, pred_y_test_Volterra, y_test], file_path)
    print("The variable 'data' has been saved successfully.")
print(krCV_Volterra.best_params_)
print(krCV_Volterra.best_score_)
print('Volerra done')
%%
############## SIG TRUNCATED kernel ################
####################################################
# truncated signature kernel
est_string = 'SigTrunc'
kr = SigTruncKernel(ifscale_y=False)
pipe_kr_SigTrunc = Pipeline(
    [('scl', StandardScaler()),
    ('kr_est', kr)]
)

loc_log_alpha = -10
scale_log_alpha = -2
loc_log_gamma = -3
scale_log_gamma = 1
loc_log_scale = -1
scale_log_scale = 2
loc_log_prescale = -1
scale_log_prescale = 0

num_params = 3
num_degree = 8
n_iter = 7000

# Start the timer
start_time = time.time()

if ifRandomSearch:
    grid_param_SigTrunc = dict(kr_est__alpha=loguniform(a=10**loc_log_alpha, b=10**scale_log_alpha),
                             kr_est__gamma=loguniform(a=10**loc_log_gamma, b=10**scale_log_gamma),
                             kr_est__depth=[2, 3, 4, 5, 6, 7, 8, 9],
                             kr_est__scale=loguniform(a=10**loc_log_scale, b=10**scale_log_scale),
                             kr_est__prescale=loguniform(a=10**loc_log_prescale, b=10**scale_log_prescale),
                             kr_est__kernel= ['linear', 'rbf'],
                             kr_est__normalize= [False, True])
    krCV_SigTrunc = RandomizedSearchCV(
        estimator=pipe_kr_SigTrunc,
        param_distributions=grid_param_SigTrunc, 
        cv=TimeSeriesSplit(n_splits=5), 
        n_jobs=n_jobs,
        scoring=my_scorer,
        refit=my_refit,
        random_state=0,
        n_iter = n_iter
    )
else:
    grid_param_SigTrunc = {
        'kr_est__alpha': np.logspace(loc_log_alpha, scale_log_alpha, int((n_iter/num_degree)**(num_params**-1))),
        'kr_est__gamma': np.append(1, np.logspace(loc_log_gamma, scale_log_gamma, int((n_iter/num_degree)**(num_params**-1))-1)),
        'kr_est__depth': [2,3],
        'kr_est__prescale': np.append(1, np.logspace(loc_log_prescale, scale_log_prescale, int((n_iter/num_degree)**(num_params**-1))-1)),
        'kr_est__kernel': ['linear', 'rbf'],
        'kr_est__normalize': [False, True]
    }
    krCV_SigTrunc = GridSearchCV(
        estimator=pipe_kr_SigTrunc,
        param_grid=grid_param_SigTrunc, 
        cv=TimeSeriesSplit(n_splits=5), 
        n_jobs=n_jobs,
        scoring=my_scorer,
        refit=my_refit
    )


krCV_SigTrunc.fit(x_train, y_train_demeaned)
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print("RandomizedSearchCV took {:.2f} seconds.".format(elapsed_time))

score = krCV_SigTrunc.cv_results_[score_str]
print(pd.DataFrame(krCV_SigTrunc.cv_results_).sort_values(by=score_str))
pred_y_test_SigTrunc=krCV_SigTrunc.predict(x_test)
pred_y_test_SigTrunc = scaler_y.inverse_transform(pred_y_test_SigTrunc)

nmsekr_SigTrunc = perf.calculate_nmse(y_test, pred_y_test_SigTrunc)
# Print testing normalized mean squared error
print("Testing NMSE: ", nmsekr_SigTrunc)
plt.plot(pred_y_test_SigTrunc[0:100,1])
plt.plot(y_test[0:100,1])

time_string = 'mean_'+ set_my_scorer+'_time'
mean_fit_time= krCV_SigTrunc.cv_results_['mean_fit_time']
mean_est_time= krCV_SigTrunc.cv_results_[time_string]
n_splits  = krCV_SigTrunc.n_splits_ #number of splits of training data
n_iter = pd.DataFrame(krCV_SigTrunc.cv_results_).shape[0] #Iterations per split

print(np.mean(mean_fit_time + mean_est_time) * n_splits * n_iter)

if ifsave:
    # Saving Variables
    file_path = 'data_score_' + str(ndim)+'d_'+set_my_scorer+ '_'+ est_string+ '_'+ '.pickle'

    # write variables to filename
    hkl.dump([krCV_SigTrunc.best_estimator_, krCV_SigTrunc.best_params_,
                     krCV_SigTrunc.best_score_,
                    nmsekr_SigTrunc, elapsed_time, pred_y_test_SigTrunc, y_test, n_iter, loc_log_alpha, 
                    scale_log_alpha, loc_log_gamma, scale_log_gamma, loc_log_scale, scale_log_scale, 
                    loc_log_prescale, scale_log_prescale], file_path)
    print("The variable 'data' has been saved successfully.")

print(krCV_SigTrunc.best_params_)
print('SigTrunc done')

# %%
################ GAK kernel ##########################
######################################################
# gak kernel
# Define a custom kernel function to use the precomputed GAK kernel
est_string = 'gak'
n_iter=4
kr = GAKKernel(ifscale_y=False, ifcenter_kernel = True)
pipe_kr_gak = Pipeline(
    #[('scl', StandardScaler()),
    [('kr_est', kr)]
)

loc_log_alpha = -7
scale_log_alpha = 5
loc_log_sigma = -2
scale_log_sigma = 2
num_params = 
# Start the timer
start_time = time.time()
if ifRandomSearch:
    grid_param_kr_gak = dict(kr_est__alpha=loguniform(a=10**loc_log_alpha, b=10**scale_log_alpha),
                             kr_est__sigma=loguniform(a=10**loc_log_sigma, b=10**scale_log_sigma))
    krCV_gak = RandomizedSearchCV(
        estimator=pipe_kr_gak,
        param_distributions=grid_param_kr_gak, 
        cv=TimeSeriesSplit(n_splits=5), 
        n_jobs=n_jobs,
        scoring=my_scorer,
        refit=my_refit,
        random_state=0,
        n_iter = n_iter
    )
else:
    grid_param_kr_gak = {
        'kr_est__alpha': np.append(0, np.logspace(loc_log_alpha, scale_log_alpha, int((n_iter)**(num_params**-1))-1)),
        'kr_est__sigma': np.append(1, np.logspace(loc_log_sigma, scale_log_sigma, int((n_iter)**(num_params**-1))-1))
    }
    krCV_gak = GridSearchCV(
        estimator=pipe_kr_gak,
        param_grid=grid_param_kr_gak, 
        cv=TimeSeriesSplit(n_splits=5), 
        n_jobs=n_jobs,
        scoring=my_scorer,
        refit=my_refit
    )
    

krCV_gak.fit(x_train, y_train_demeaned)
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print("RandomizedSearchCV took {:.2f} seconds.".format(elapsed_time))

score = krCV_gak.cv_results_[score_str]
print(pd.DataFrame(krCV_gak.cv_results_).sort_values(by=score_str))
pred_y_test_gak=krCV_gak.predict(x_test)
pred_y_test_gak = scaler_y.inverse_transform(pred_y_test_gak)

nmsekr_gak = perf.calculate_nmse(y_test, pred_y_test_gak)
# Print testing normalized mean squared error
print("Testing NMSE: ", nmsekr_gak)
plt.plot(pred_y_test_gak[0:t_dispay,1])
plt.plot(y_test[0:t_dispay,1])         
               
# GAK verification
# #####################################################
# sigma=10
# kk = gak_kernel_callable(x_train, x_train, sigma=sigma)
# nkk = kk.shape[0]
# # #alpha_ols = np.matmul(np.linalg.pinv((Gram)), target_data)
# reg = 1e-12
# alpha_ols = np.matmul(np.linalg.inv((kk + reg * np.identity(nkk))), y_train)
# alpha0e_ols = np.mean(y_train, axis=0) - np.mean(np.matmul(alpha_ols.transpose(), kk).transpose(), axis=0)

# # Gram_new = np.vstack([kk, np.ones(len(kk))]).T
# # R, residuals, RANK, sing = lstsq(Gram_new, y_train)
# # alpha_ols = R[0:np.shape(R)[0]-1,0:]
# # alpha0_ols = R[np.shape(R)[0]-1,0:]
# alphas_int = alpha0e_ols

# pred_block_insample = np.matmul(alpha_ols.transpose(), kk)
# for t in range(nkk):
#     pred_block_insample[0:, t] = pred_block_insample[0:, t]+ alphas_int
    
# pred_block_insample = pred_block_insample.transpose()
# nmsekr = perf.calculate_nmse(pred_block_insample, y_train)
# print("Training NMSE: ", nmsekr)
# kk_test = gak_kernel_callable(x_test, x_train, sigma=sigma)
# nkk_test = kk_test.shape[0]
# pred_y_test = np.matmul(alpha_ols.transpose(), kk_test.transpose())
# for t in range(nkk_test):
#     pred_y_test[0:, t] = pred_y_test[0:, t]+ alphas_int
    
# pred_y_test = pred_y_test.transpose()
# nmsekr = perf.calculate_nmse(pred_y_test, y_test)

#####################################################
time_string = 'mean_'+ set_my_scorer+'_time'
mean_fit_time= krCV_gak.cv_results_['mean_fit_time']
mean_est_time= krCV_gak.cv_results_[time_string]
n_splits  = krCV_gak.n_splits_ #number of splits of training data
n_iter = pd.DataFrame(krCV_gak.cv_results_).shape[0] #Iterations per split

print(np.mean(mean_fit_time + mean_est_time) * n_splits * n_iter)

if ifsave:
    # Saving Variables
    file_path = 'data_score_' + str(ndim)+'d_'+set_my_scorer+ '_'+ est_string+ '_'+ '.pickle'

    # write variables to filename
    hkl.dump([krCV_gak.best_estimator_, krCV_gak.best_params_,
                     krCV_gak.best_score_,
                    nmsekr_gak, elapsed_time, pred_y_test_gak, y_test], file_path)
    print("The variable 'data' has been saved successfully.")

print(krCV_gak.best_params_)
print('gak done')

%%
########### SIG TRUNCATED EXPLICIT kernel ###########
#####################################################
truncated signature kernel
x_train_s = sigkernel.transform(x_train, at=True, ll=True, scale=.1)
x_test_s = sigkernel.transform(x_test, at=True, ll=True, scale=.1)
best_error_sig = 1e8
# truncated signature kernel
grid_param_truncsign = {
    'kr_est__alpha': np.logspace(-5, 5, 5),
    'kr_est__gamma': np.logspace(-5, 5, 5)
}

for depth in [2, 3, 4,5]:#tqdm([2, 3, 4, 5, 6]):
    for scale in np.logspace(-1, 2, 5):#tqdm(np.logspace(-1, 2, 20), leave=False):
        for ker in ['linear', 'rbf']:#tqdm(['linear', 'rbf'], leave=False):
            for normalize in [True, False]:#tqdm([True, False], leave=False):
                kr = KernelRidge(kernel=ker)
                # pipe_kr = Pipeline([('scl', StandardScaler()),  
                #         ('kr_est', kr)])
                pipe_kr = Pipeline([('kr_est', kr)])
                # truncated signatures
            
                sig_train = iisignature.sig(scale*x_train_s, depth)
            
                # normalization
                if normalize:
                    sig_train = sigkernel.normalize(sig_train, x_train_s.shape[-1], depth)


                # fit the model
                krCV = GridSearchCV(
                    estimator=pipe_kr,
                    param_grid=grid_param_truncsign, 
                    cv=TimeSeriesSplit(n_splits=5), n_jobs=-1
                )
                
                krCV.fit(sig_train, y_train)
                

                # select best model (criterion: R^2)
                if np.abs(1.-krCV.best_score_) < np.abs(1.-best_error_sig):
                    best_sig_model = krCV
                    best_error_sig = krCV.best_score_
                    best_depth_sig = depth
                    best_scale_sig = scale
                    best_ker_sig = ker
                    normalize_sig = normalize
                    #print(pd.DataFrame(best_sig_model.cv_results_).sort_values(by='mean_test_score'))
                    pred_y_train=krCV.predict(sig_train)
                    nmsekr = perf.calculate_nmse(pred_y_train, y_train)
                    print("Testing NMSE: ", nmsekr)

sig_test = iisignature.sig(best_scale_sig*x_test_s, best_depth_sig)
# normalization
if normalize_sig:
    sig_train = sigkernel.normalize(sig_test, x_test_s.shape[-1], best_depth_sig)
pred_y_test=best_sig_model.predict(sig_test)
nmsekr = perf.calculate_nmse(pred_y_test, y_test)
# Print testing normalized mean squared error
print("Testing NMSE: ", nmsekr)
plt.plot(pred_y_test[0:100,1])
plt.plot(y_test[0:100,1])

# # %%

# %%
# # %%
# import hickle as hkl
# est_string='gak'
# ndim=5
# set_my_scorer = 'score'
# file_path = 'data_score_' + str(ndim)+'d_'+set_my_scorer+ '_'+ est_string+ '_'+ '.pickle'
# array_results=hkl.load(file_path)
# # %%
# print(array_results[0])
# print(array_results[1])
# print(array_results[2])
# print(array_results[3])
# print(array_results[4])
# print(array_results[5])
# %%
# import matplotlib.pyplot as plt
# import hickle as hkl
# est_string='Volterra'
# ndim=5
# set_my_scorer='score'
# file_path = 'data_score_' + str(ndim)+'d_'+set_my_scorer+ '_'+ est_string+ '_'+ '.pickle'
# array_results=hkl.load(file_path)
# yt1=array_results[6]
# yt=array_results[7]
# plt.plot(yt1[:,0])
# plt.plot(yt[:,0])
# from sklearn.metrics import median_absolute_error
# m=median_absolute_error(yt,yt1)

# from scipy.stats import wasserstein_distance
# import numpy as np
# ndim=5
# was_dist=[]
# for i in range(15):
#     was_dist = np.append(was_dist, wasserstein_distance(yt[:,i], yt1[:,i]))
# ws=np.mean(was_dist)
# ws