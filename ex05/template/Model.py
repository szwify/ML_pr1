# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:48:53 2018
@author: Sacha
"""
import numpy as np

def compute_error(y, tx, w):
    y = y.reshape((-1,1))
    return y - tx.dot(w)

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)
	
def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def calculate_sig_mse(y, tx, w):
    """Calculate the sigmoid mse for vector e."""
    return np.mean(np.log(1+np.exp(tx.dot(w)))-y*tx.dot(w))
	

def subtract_by_part(x1, x2, n):
    l = len(x1)
    ln = int(l/n)
    sub = x1[0:n]-x2[0:n]
    if l<n : return sub
    for i in range(1, ln):
        temp = x1[i*n:(i+1)*n]-x2[i*n:(i+1)*n]
        sub = np.hstack((sub,temp))
    temp = x1[ln*n:]-x2[ln*n:]
    sub = np.hstack((sub,temp))
    return sub
    
def compute_gradient_mse( y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err
    
def compute_gradient_mae( y, tx, w):
    """Compute the gradient."""
    print('Gradient MAE not impleted')
    return None
	
def compute_gradient_sig_mse(y, tx, w):
    err = y - sigmoid(tx.dot(w))
    return tx.T.dot(-err), err

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************  
    return np.mean(np.log(1+np.exp(tx.dot(w)))-y*tx.dot(w))
    
def choose_function(loss_function):
    if loss_function == 'mse':
        return compute_gradient_mse, calculate_mse
    if loss_function == 'mae':
        return compute_gradient_mae, calculate_mae
    if loss_function == 'sig_mse':
        return compute_gradient_sig_mse, calculate_sig_mse


def sigmoid(t):
    """apply sigmoid function on t."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    return np.exp(t) / (1+np.exp(t))

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate hessian: TODO
    # ***************************************************
    S = np.zeros((len(tx), len(tx)))
    for row in range (0, len(tx)):
        S[row, row] = sigmoid(tx[row].dot(w))*(1-sigmoid(tx[row].dot(w)))
    return tx.T.dot(S.dot(tx))


    

def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient, and hessian: TODO
    # ***************************************************
    hessian = calculate_hessian(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    loss = calculate_loss(y, tx, w)
    return loss, grad, hessian

def learning_by_newton_method(y, tx, w):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient and hessian: TODO
    # ***************************************************
    loss, grad, hessian = logistic_regression(y, tx, w)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # update w: TODO
    # ***************************************************
    w = w-np.linalg.inv(hessian).dot(grad)
    return loss, w

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
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
     
    def __init__(self, max_iters = 100, gamma = 1, lambda_ = 1, loss_function = 'mse'):
        self.max_iters = max_iters
        self.gamma = gamma
        self.lambda_ = lambda_
        self.compute_gradient, self.compute_loss = choose_function(loss_function)
        
    
        
    def fit(self, y, tx):
        """Stochastic gradient descent."""
        # Define parameters to store w and loss
        # Use the standard mini-batch-size 1
        gamma = self.gamma
        lambda_ = self.lambda_
        compute_gradient, compute_loss = self.compute_gradient, self.compute_loss
        
        batch_size = 1
        initial_w = np.zeros((len(tx[0]),1))
        ws = [initial_w]
        losses = []
        w = initial_w
        for n_iter in range(self.max_iters):
            for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
                # compute a stochastic gradient and loss
                grad, err = compute_gradient(y_batch, tx_batch, w)
                # update w through the stochastic gradient update
                w = w*(1-2*gamma*lambda_) - gamma * grad
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
        self.compute_gradient, self.compute_loss = choose_function('mse')
        
    def fit(self, y,tx):
        aI = self.lambda_ * np.identity(tx.shape[1])
        a = tx.T.dot(tx) + aI
        b = tx.T.dot(y)
        self.w_ = np.linalg.solve(a, b)
        self.losses_ = self.compute_loss(y - tx.dot(self.w_))

    def predict(self, tx):
        return tx.dot(self.w_)

class logistic_regression_Newton(Model):
    def __init__(self, max_iters = 100, treshold = 1e-8):
        self.max_iters = max_iters
        self.treshold = treshold
        
    def fit(self, y, tx):
        # init parameters
        max_iters = self.max_iters
        threshold = self.treshold
        
        initial_w = np.zeros((len(tx[0]),1))
        ws = [initial_w]
        losses = []
        w = initial_w
    
        # start the logistic regression
        for e in range(max_iters):
            # get loss and update w.
            loss, w = learning_by_newton_method(y, tx, w)
            # log info
            # converge criterion
            losses.append(loss)
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break
            
        self.losses_ = losses
        self.w_ = w
        return None
    
    def predict(self, tx):
        return tx.dot(self.w_)
    
class logistic_regression_gradient(Model):
    def __init__(self, gamma = 0.001, max_iters = 100, treshold = 1e-8):
        self.max_iters = max_iters
        self.treshold = treshold
        self.gamma = gamma
        
    def fit(self, y, tx):
        # init parameters
        max_iters = self.max_iters
        threshold = self.treshold
        gamma = self.gamma
        initial_w = np.zeros((len(tx[0]),1))
        ws = [initial_w]
        losses = []
        w = initial_w
    
        # start the logistic regression
        for e in range(max_iters):
            # get loss and update w.
            loss, w = learning_by_gradient_descent(y, tx, w, gamma)
            # log info
            # converge criterion
            losses.append(loss)
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break
            
        self.losses_ = losses
        self.w_ = w
        return None
    
    def predict(self, tx):
        return tx.dot(self.w_)
    
class logistic_regression_SGD(Model):
    def __init__(self, gamma = 0.001, lambda_ = 0.01, batch_size = 1000, max_iters = 5, treshold = 1e-8, loss_function = 'sig_mse'):
        self.max_iters = max_iters
        self.treshold = treshold
        self.gamma = gamma
        self.batch_size = batch_size
        self.lambda_ = lambda_
        self.compute_gradient, self.compute_loss = choose_function(loss_function)
        
    def fit(self, y, tx):
        # init parameters
        compute_gradient, compute_loss = self.compute_gradient, self.compute_loss
        max_iters = self.max_iters
        threshold = self.treshold
        batch_size = self.batch_size
        lambda_ = self.lambda_
        gamma = self.gamma
        initial_w = np.zeros((len(tx[0]),1))
        losses = []
        w = initial_w
        end = False
        # start the logistic regression
        for e in range(max_iters):
            for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
                
                # get loss and update w.
				
                
                grad, _ = compute_gradient(y_batch, tx_batch, w)
                
                w = (1-2*gamma*lambda_)*w-gamma*grad
                loss = compute_loss(y_batch, tx_batch,w)
                
                losses.append(loss)
            if end :
                break
            
        self.losses_ = losses
        self.w_ = w
        return None
    
    def predict(self, tx):
        return tx.dot(self.w_)