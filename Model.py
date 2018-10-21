# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:48:53 2018

@author: Sacha
"""
import numpy as np

class Model:
    
    
        
    def calculate_mse(self, e):
        """Calculate the mse for vector e."""
        return 1/2*np.mean(e**2)
    
    def calculate_mae(self, e):
        """Calculate the mae for vector e."""
        return np.mean(np.abs(e))
    
    
    def compute_loss(self, y, tx, w):
        """Calculate the loss using mse"""
        e = y - tx.dot(w)
        
        if self.loss_function_ == 'mse':
            return self.calculate_mse(e)
        if self.loss_function_ == 'mae':
            return self.calculate_mse(e)
        
        
    def compute_gradient_mse(self, y, tx, w):
        """Compute the gradient."""
        err = y - tx.dot(w)
        grad = -tx.T.dot(err) / len(err)
        return grad, err
        
    def compute_gradient_mae(self, y, tx, w):
        """Compute the gradient."""
        print('Gradient MAE not impleted')
        return None
    
    def compute_gradient(self, y, tx, w):
        if self.loss_function_ == 'mse':
            return self.compute_gradient_mse(y, tx, w)
        if self.loss_function_ == 'mae':
            return self.compute_gradient_mae(y, tx, w)
            


        
class least_squares_GD(Model):
     
    def __init__(self, max_iters = 100, gamma = 1, loss_function = 'mse'):
        self.max_iters = max_iters
        self.gamma = gamma
        self.loss_function_ = loss_function
        
    def fit(self, y, tx):
        """Gradient descent algorithm."""
        # Define parameters to store w and loss
        initial_w = np.zeros((len(tx[0]),1))
        ws = [initial_w]
        losses = []
        w = initial_w
        for n_iter in range(self.max_iters):
            # compute loss, gradient
            grad, err = self.compute_gradient(y, tx, w)
            loss = self.calculate_mse(err)
            # gradient w by descent update
            w = w - self.gamma * grad
            # store w and loss
            ws.append(w)
            losses.append(loss)
        self.losses_ = losses
        self.w_ = w
        return None
    
    def predict(self, tx):
        return tx.dot(self.w_)
        
    
        