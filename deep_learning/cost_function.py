import numpy as np

def cost_function(A2, Y):
    """
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    cost 
    (simply, we can use this variable to measure the performance of the machine)
    """

    loss = -(Y*np.log(A2) + (1-Y)*np.log(1-A2))
    cost = np.sum(loss)/len(Y)
    return(cost)
