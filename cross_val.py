# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:13:50 2018

@author: Sacha
"""

import numpy as np

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

        model.fit(x_tr, y_tr)
        loss_test = model.compute_loss(x_te, y_te, model.w_)
        loss_train = model.compute_loss(x_tr, y_tr, model.w_)
        
        losses_test.append(loss_test)
        losses_train.append(loss_train)
        
    return losses_train, losses_test

