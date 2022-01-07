import numpy as np

def initialize(n_x, n_h, n_y):
    W1 = np.random.rand(n_h,n_x)*0.0001
    b1 = np.zeros((n_h,1))
    W2 = np.random.rand(n_y, n_h)*0.01
    b2 = np.zeros((n_y,1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


