# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:48:53 2018

@author: Sacha
"""
import numpy as np

def compute_error(y, tx, w):
    return y - tx.dot(w)

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))
    
    
def compute_gradient_mse( y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err
    
def compute_gradient_mae( y, tx, w):
    """Compute the gradient."""
    print('Gradient MAE not impleted')
    return None

    
def choose_function(loss_function):
    if loss_function == 'mse':
        return compute_gradient_mse, calculate_mse
    if loss_function == 'mae':
        return compute_gradient_mae, calculate_mae
        
class Model:
    def __init__(self, args):
        return
        

            


        
class least_squares_GD(Model):
     
    def __init__(self, max_iters = 100, gamma = 1, loss_function = 'mse'):
        self.max_iters = max_iters
        self.gamma = gamma
        self.compute_gradient, self.compute_loss = choose_function(loss_function)
        
    def fit(self, y, tx):
        """Gradient descent algorithm."""
        # Define parameters to store w and loss
        gamma = self.gamma
        compute_gradient, compute_loss = self.compute_gradient, self.compute_loss
        
        initial_w = np.zeros((len(tx[0]),1))
        ws = [initial_w]
        losses = []
        w = initial_w
        
        for n_iter in range(self.max_iters):
            # compute loss, gradient
            grad, err = compute_gradient(y, tx, w)
            loss = compute_loss(err)
            # gradient w by descent update
            w = w - gamma * grad
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
        self.compute_gradient, self.compute_loss = choose_function(loss_function)
        
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
        gamma = self.gamma
        compute_gradient, compute_loss = self.compute_gradient, self.compute_loss
        
        batch_size = 1
        initial_w = np.zeros((len(tx[0]),1))
        ws = [initial_w]
        losses = []
        w = initial_w
        for n_iter in range(self.max_iters):
            for y_batch, tx_batch in self.batch_iter(y, tx, batch_size=batch_size, num_batches=1):
                # compute a stochastic gradient and loss
                grad, err = compute_gradient(y_batch, tx_batch, w)
                # update w through the stochastic gradient update
                w = w - gamma * grad
                # calculate loss
                loss = compute_loss(err)
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
    

        
    
        