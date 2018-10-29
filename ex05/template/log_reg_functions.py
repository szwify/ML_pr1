#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


####### IMPLEMENT SIGMOID FUNCTION
def sigmoid(t):
    return 1.0/(1.0+np.exp(-t))
#######


####### COMPUTE THE LOSS WITHOUT REGULARIZATION
def log_reg_loss(y, tx, w):
    loss=np.sum(np.log(1+np.exp(tx.dot(w))))-y.T.dot(tx.dot(w))
    return loss
#######


####### COMPUTE THE LOSS WITH THE REGULARIZATION PARAMETER
def calculate_reg_loss(y, tx, w, lambda_):
    return log_reg_loss(y, tx, w) +lambda_ *w.T.dot(w)
#######
  

####### COMPUTE THE GRADIENT 
def compute_gradient_sig(y, tx, w):
    pred = sigmoid(tx.dot(w))
    return tx.T.dot(pred - y)/np.linalg.norm(pred-y)
#######


####### COMPUTE THE HESSIAN MATRIX
def compute_hessian(y, tx, w):
    pred=sigmoid(tx.dot(w))
    vect_S=pred*(1-pred)
    M_1=np.zeros(tx.shape)
    for i in range(M_1.shape[0]):
        M_1[i,:]=vect_S[i]*tx[i,:]
    return tx.T.dot(M_1)
#######
    

####### COMPUTES THE DIFFERENT FEATURES NEEDED TO THE EVOLUTION OF W
def penalized_logistic_regression(y, tx, w, lambda_):
    loss = calculate_reg_loss(y, tx, w,lambda_)
    print(compute_gradient_sig(y, tx, w).shape)
    gradient = compute_gradient_sig(y, tx, w) + 2.0 * lambda_ * w
    return loss, gradient
#######
    

####### COMPUTES THE DIFFERENT FEATURES NEEDED TO THE EVOLUTION OF W
def penalized_logistic_regression_hess(y, tx, w, lambda_):
    loss = calculate_reg_loss(y, tx, w,lambda_)
    gradient = compute_gradient_sig(y, tx, w) + 2.0 * lambda_ * w
    hessian =compute_hessian(y, tx, w)
    return loss, gradient, hessian
#######
    

####### COMPUTES THE NEW W
def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient
    return loss, w
#######


####### COMPUTES THE NEW W
def learning_by_penalized_gradient_hess(y, tx, w, gamma, lambda_):
    loss, gradient, hessian = penalized_logistic_regression_hess(y, tx, w, lambda_)
    w = w - gamma * np.linalg.inv(hessian).dot(gradient)
    return loss, w
#######
    

####### COMPUTES THE WHOLE EVOLUTION
def reg_log_reg(y,tx,step_lin,switch_dif,step_hess,tol,lambda_):
    w=np.zeros(tx.shape[1]) #INITIAL W
    n_iters=0
    last_loss=0
    last_w=np.zeros((tx.shape[1], 1))
    loss=0
    while ((n_iters==0 or abs(last_loss-loss)>tol) and not np.isinf(loss)):
        if (n_iters%50==0):
            print(n_iters)
            print(loss)
        if (abs(last_loss-loss)>= switch_dif or n_iters == 0): # IF THE LINEAR STEP EVOLUTION IS EFFICIENT
            last_w=w
            last_loss=loss
            loss,w=learning_by_penalized_gradient(y,tx,w,step_lin,lambda_)
        else:                                                  # IF THE LINEAR STEP EVOLUTION IS NOT EFFICIENT
                                                               # SWITCH TO HESSIAN STEP
            last_w=w
            last_loss=loss
            loss,w=learning_by_penalized_gradient_hess(y, tx, w, step_hess, lambda_)
        n_iters=n_iters+1

    if np.isinf(loss):
        return loss,last_w
    else:
        return loss,w
#######