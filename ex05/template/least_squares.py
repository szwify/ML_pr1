# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    print("y.shape = ", y.shape)
    print("tx.shape = ", tx.shape)
    e = y - tx.dot(w)
    print("e.shape = ", e.shape)
    mse = e.T.dot(e) / (2 * len(e))
    return mse

def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w_opt = np.linalg.solve(a, b)
    mse = compute_mse(y, tx, w_opt)
    return mse, w_opt

