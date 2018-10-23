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
            loss = self.compute_loss(y, tx, w)
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
 
    
class least_squares_SGD(Model):
     
    def __init__(self, max_iters = 100, gamma = 1, loss_function = 'mse'):
        self.max_iters = max_iters
        self.gamma = gamma
        self.loss_function_ = loss_function
        
    def batch_iter(self, y, tx, batch_size, num_batches=1, shuffle=True):
        """
        Generate a minibatch iterator for a dataset.
        Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
        Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
        Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
        Example of use :
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
            <DO-SOMETHING>
        """
        data_size = len(y)
    
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_y = y[shuffle_indices]
            shuffled_tx = tx[shuffle_indices]
        else:
            shuffled_y = y
            shuffled_tx = tx
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if start_index != end_index:
                yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
        
    def fit(self, y, tx):
        """Stochastic gradient descent."""
        # Define parameters to store w and loss
        # Use the standard mini-batch-size 1
        batch_size = 1
        initial_w = np.zeros((len(tx[0]),1))
        ws = [initial_w]
        losses = []
        w = initial_w
        for n_iter in range(self.max_iters):
            for y_batch, tx_batch in self.batch_iter(y, tx, batch_size=batch_size, num_batches=1):
                # compute a stochastic gradient and loss
                grad, _ = self.compute_gradient(y_batch, tx_batch, w)
                # update w through the stochastic gradient update
                w = w - self.gamma * grad
                # calculate loss
                loss = self.compute_loss(y, tx, w)
                # store w and loss
                ws.append(w)
                losses.append(loss)
        
        self.losses_ = losses
        self.w_ = w
        return None
    
    def predict(self, tx):
        return tx.dot(self.w_)
    
    

class ridge_regression(Model):
     
    def __init__(self, lamb = 1):
        self.lambda_ = lamb
        self.loss_function_ = 'mse'
        
    def fit(self, y,tx):
        aI = self.lambda_ * np.identity(tx.shape[1])
        a = tx.T.dot(tx) + aI
        b = tx.T.dot(y)
        self.w_ = np.linalg.solve(a, b)
        self.losses_ = self.compute_loss(y, tx, self.w_)

    def predict(self, tx):
        return tx.dot(self.w_)
    

        
    
        