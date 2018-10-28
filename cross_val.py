# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:13:50 2018

@author: Sacha
"""

import numpy as np
from Model import *
import matplotlib.pyplot as plt

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(model, y, x, k_fold, seed = 1):
    """return the loss of ridge regression."""
    k_indices = build_k_indices(y, k_fold, seed)
    losses_test = []
    losses_train = []
    for k in range(k_fold):
        selector_tr = [row for row in range(k_indices.shape[0]) if row != k]
        index_tr = k_indices[selector_tr].flatten()

        index_te = k_indices[k].flatten()

        x_te = x[index_te]
        y_te = y[index_te]
        x_tr = x[index_tr]
        y_tr = y[index_tr]

        model.fit(y_tr, x_tr)
        err_te = compute_error(y_te, x_te, model.w_)
        err_tr = compute_error(y_tr, x_tr, model.w_)
        
        loss_te = np.sqrt(2*model.compute_loss(err_te))
        loss_tr = np.sqrt(2*model.compute_loss(err_tr))
        
        losses_test.append(loss_te)
        losses_train.append(loss_tr)
        
    return losses_train, losses_test

def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
