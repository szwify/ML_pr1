# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

def compute_mse(y, tx, w):
    e = y - tx.T.dot(w)
    return np.mean(e.dot(e)) / 2