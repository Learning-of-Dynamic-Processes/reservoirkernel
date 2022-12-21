#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 13:03:48 2022

@author: Lukas Gonon, Lyudmila Grigoryeva, and Juan-Pablo Ortega
"""
import signal
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.metrics import mean_absolute_error
import scipy.io as spio
from sklearn.svm import SVR

import sigkernel

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
          'figure.figsize': (16, 5),
          'axes.labelsize': 'large',
          'axes.titlesize':'large',
          'xtick.labelsize':'large',
          'ytick.labelsize':'large'}
pylab.rcParams.update(params)


class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):  # Custom signal handler
    raise TimeoutException

def mean_absolute_percentage_error(y_true, y_pred):
    return 100.*np.mean(np.abs((y_true - y_pred) / y_true))

def absolute_percentage_error(y_true, y_pred):
    len_ape = np.shape(y_true)[0];
    ape = np.zeros((len_ape, 1))
    for i in range(len_ape):
        ape[i] = 100.*np.mean(np.abs((y_true[i] - y_pred[i]) / y_true[i]))
    return ape

#Helper function that extract rolling windows of historical prices of size h and means of the next future f prices.
def GetWindow(x, h_window=30, f_window=10):

    # First window
    X = np.array(x.iloc[:h_window,]).reshape(1,-1)

    # Append next window
    for i in range(1,len(x)-h_window+1):
        x_i = np.array(x.iloc[i:i+h_window,]).reshape(1,-1)
        X = np.append(X, x_i, axis=0)

    # Cut the end that we can't use to predict future price
    rolling_window = (pd.DataFrame(X)).iloc[:-f_window,]
    return rolling_window

def GetNextMean(x, h_window=30, f_window=10):
    return pd.DataFrame((x.rolling(f_window).mean().iloc[h_window+f_window-1:,]))

def PlotResult(y_train, y_test, y_train_predict, y_test_predict, name):

    train_len = len(y_train)
    test_len = len(y_test)

    #Visualise
    fig, ax = plt.subplots(1, figsize=(12, 5))
    ax.plot(y_train_predict,color='red')

    ax.plot(range(train_len, train_len+test_len),
            y_test_predict,
            label='Predicted average price',
            color='red',linestyle = '--')

    ax.plot(np.array((y_train).append(y_test)),
             label='Actual average price',
             color='green')

    ax.axvspan(len(y_train), len(y_train)+len(y_test),
                alpha=0.3, color='lightgrey')

    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc="best")
    plt.xlabel('Days')
    plt.ylabel('Bitcoin prices')
    plt.savefig('../pictures/bitcoin_prices_prediction_{}'.format(name))
    plt.show()


def PlotResultCummean(y_sigPDE, y_reservoir, name):

    er_len = len(y_sigPDE)

    #Visualise
    fig, ax = plt.subplots(1, figsize=(12, 5))
    #ax.plot(y_reservoir,color='red')
    ax.plot(range(0, er_len),
            y_sigPDE,
            label='PDE average price',
            color='blue')
    
    ax.plot(range(0, er_len),
            y_reservoir,
            label='Res average price',
            color='red')

    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc="best")
    plt.xlabel('Days')
    plt.ylabel('Bitcoin prices')
    plt.savefig('../pictures/bitcoin_prices_prediction_{}'.format(name))
    plt.show()
    
    
    
# load data (source is https://www.cryptodatadownload.com)
BTC_price = pd.read_csv('data/gemini_BTCUSD_day.csv',header=1)

# drop the first column and reverse order
BTC_price = BTC_price.iloc[1:,:]
BTC_price = BTC_price.iloc[::-1]
BTC_price['Date'] = pd.to_datetime(BTC_price['Date'])
BTC_price.set_index('Date', inplace=True)

# select duration
initial_date = '2017-06-01'
finish_date = '2018-08-01'
BTC_price = BTC_price.loc[BTC_price.index >= initial_date]
BTC_price = BTC_price.loc[BTC_price.index <= finish_date]

# use only close price
close_price = BTC_price.loc[:,'Close']
# close_price = TimeSeriesScalerMeanVariance().fit_transform(close_price.values[None,:])
close_price = pd.DataFrame(np.squeeze(close_price))

# use last h_window observations to predict mean over next f_window observations
h_window = 36
f_window = 2

# next mean price
y = GetNextMean(close_price, h_window=h_window , f_window=f_window)

# normal window features
X_window = GetWindow(close_price, h_window, f_window).values
X_window = torch.tensor(X_window, dtype=torch.float64)
X_window = sigkernel.transform(X_window, at=True, ll=True, scale=1e-5)

# train test split
x_train, x_test, y_train, y_test = train_test_split(X_window, y, test_size=0.2, shuffle=False)
x_train_len = np.shape(x_train)[0]
x_test_len = np.shape(x_test)[0]
total_len = h_window - 1 + x_test_len + x_train_len
# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)

# hyperparameters for grid search
svr_parameters = {'C': np.logspace(0, 4, 5), 'gamma': np.logspace(-4, 4, 9)}
best_score = -1e8
x_train = torch.tensor(x_train, dtype=torch.float64, device='cpu')
x_test = torch.tensor(x_test, dtype=torch.float64, device='cpu')
i = 0
perf_score = 'neg_mean_absolute_percentage_error'
scores_pde = np.zeros((22, 1))
for sigma in np.array([5e-2, 1e-1, 2.5e-1, 5e-1, 7.5e-1, 1.]):
    print(sigma)
    # Specify the static kernel
    static_kernel = sigkernel.RBFKernel(sigma=sigma)

    # Initialize the corresponding signature kernel
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=0)

    # Gram matrix train
    G_train = signature_kernel.compute_Gram(x_train, x_train, sym=True).numpy()

    # fit the model    
    signal.alarm(20)
    try: 
        svr = SVR(kernel='precomputed')
        svr_pde = GridSearchCV(estimator=svr, param_grid=svr_parameters, cv=5, n_jobs=-1)#, scoring=perf_score)
        svr_pde.fit(G_train, np.squeeze(y_train))
        print(svr_pde.best_score_)
        scores_pde[i] = svr_pde.best_score_
        i = i + 1
        if svr_pde.best_score_ > best_score:
            best_pde_model = svr_pde
            best_score = svr_pde.best_score_
            best_sigma = sigma
            print(best_pde_model.best_params_['C'])
    except TimeoutException:
        continue # continue the for loop if svr takes more than 20 seconds
    else:
        # Reset the alarm
        signal.alarm(0)
                   
ker='sig_PDE'

# Specify the static kernel
static_kernel = sigkernel.RBFKernel(sigma=best_sigma)

# Initialize the corresponding signature kernel
signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=0)
# Gram matrix test
G_train = signature_kernel.compute_Gram(x_train, x_train, sym=True).numpy()
G_test = signature_kernel.compute_Gram(x_test, x_train, sym=False).numpy()

# predict
y_train_predict = best_pde_model.predict(G_train)
y_test_predict = best_pde_model.predict(G_test)

# calculate errors
p_error_test_sigPDE = mean_absolute_percentage_error(np.array(y_test).reshape(-1,1), np.array(y_test_predict).reshape(-1,1))
p_error_test_ape_sigPDE = absolute_percentage_error(np.array(y_test).reshape(-1,1), np.array(y_test_predict).reshape(-1,1))

# store final results
final = {}
final[ker] = p_error_test_sigPDE

# plot results
PlotResult(y_train, y_test, y_train_predict, y_test_predict, ker)

# # train test split
x_train, x_test, y_train, y_test = train_test_split(X_window, y, test_size=0.2, shuffle=False)

# hyperparameters for grid search
svr_parameters_res = {'C': np.logspace(4, 8, 10)}
sample = spio.loadmat('data/data_kernel_prep.mat', squeeze_me=True)
sample = sample["sample_normalized"]
M = np.max(np.abs(sample))
best_score = -1e8
tau = np.sqrt(1/(M**2))
tau = tau*0.99
k = 0
# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)
set_lambda_coef = np.arange(0.5,0.85,0.0008)
len_lambda_coef = set_lambda_coef.shape
scores_res = np.zeros((int(len_lambda_coef[0]), 1))
for lambda_coef in set_lambda_coef:
    print(lambda_coef)
    lambda_val = np.sqrt(1-(tau**2)*(M**2))*lambda_coef
    K = np.zeros((total_len,total_len))
    K0 = 1/(1-lambda_val**2)
    for i in range(total_len):
        for j in range(i+1):
            if (i == 0) or (j==0):
                K[i, j] = 1 + lambda_val**2 * K0/(1-(tau**2)*(sample[i]*sample[j]))
            else:
                K[i, j] = 1 + lambda_val**2 * K[i-1,j-1]/(1-(tau**2)*(sample[i]*sample[j]))
            K[j,i] = K[i, j]

    # Gram matrix train
    G_train = K[h_window-1:h_window+x_train_len-1,h_window-1:h_window+x_train_len-1]

    # fit the model
    signal.alarm(10)    
    
    try: 
        svr = SVR(kernel='precomputed')
        svr_reservoir = GridSearchCV(estimator=svr, param_grid=svr_parameters_res, cv=5, n_jobs=-1)#, scoring=perf_score)
        svr_reservoir.fit(G_train, np.squeeze(y_train))
        print(svr_reservoir.best_score_)
        scores_res[k] = svr_reservoir.best_score_
        k = k + 1
        if svr_reservoir.best_score_ > best_score:
        #if np.abs(1.-svr_reservoir.best_score_) < np.abs(1.-best_error):
            best_reservoir_model = svr_reservoir
            print(best_reservoir_model.best_params_['C'])
            best_score = svr_reservoir.best_score_
            best_lambda_coef = lambda_coef
    except TimeoutException:
        continue # continue the for loop if svr takes more than 10 seconds
    else:
        # Reset the alarm
        signal.alarm(0)

lambda_val = np.sqrt(1-(tau**2)*(M**2))*best_lambda_coef
K = np.zeros((total_len,total_len))
K0 = 1/(1-lambda_val**2)
for i in range(total_len):
    for j in range(i+1):
        if (i == 0) or (j==0):
            K[i, j] = 1 + lambda_val**2 * K0/(1-(tau**2)*(sample[i]*sample[j]))
        else:
            K[i, j] = 1 + lambda_val**2 * K[i-1,j-1]/(1-(tau**2)*(sample[i]*sample[j]))
        K[j,i] = K[i, j]

# Gram matrix train and test
G_train = K[h_window-1:x_train_len+h_window-1,h_window-1:x_train_len+h_window-1]
G_test = K[x_train_len+h_window-1:,h_window-1:x_train_len+h_window-1]

best_lambda_coef_res=best_lambda_coef

# fit the model
svr_reservoir = best_reservoir_model
svr_reservoir.fit(G_train, np.squeeze(y_train))

# predict
y_train_predict = svr_reservoir.predict(G_train)
y_test_predict = svr_reservoir.predict(G_test)

# calculate errors
p_error_test_reservoir = mean_absolute_percentage_error(np.array(y_test).reshape(-1,1), np.array(y_test_predict).reshape(-1,1))
p_error_test_ape_reservoir = absolute_percentage_error(np.array(y_test).reshape(-1,1), np.array(y_test_predict).reshape(-1,1))

# store final results
ker = 'reservoir'
final[ker] = p_error_test_reservoir

# plot results
PlotResult(y_train, y_test, y_train_predict, y_test_predict, ker)
   
PlotResultCummean(np.cumsum(p_error_test_ape_reservoir), np.cumsum(p_error_test_ape_sigPDE), 'comparison') 
#/ np.arange(1,79)
