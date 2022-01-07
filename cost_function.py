import numpy as np

def cost_function(A2, Y):

    loss = -(Y*np.log(A2) + (1-Y)*np.log(1-A2))
    cost = np.sum(loss)/len(Y)
    return(cost)
