import numpy as np

def initialize(n_x, n_h, n_y):
    """ 
    Arguments: 
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing parameters:
                    hidden layer 1:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    hidden layer 2:
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    # 0.01 is used to decrease Z1, Z2 (the tanh and sigmoid function may get very close to 1 if the 
    # weight is too large, which reduce the "learning speed")
    W1 = np.random.rand(n_h,n_x)*0.01   
    b1 = np.zeros((n_h,1))
    W2 = np.random.rand(n_y, n_h)*0.01
    b2 = np.zeros((n_y,1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


