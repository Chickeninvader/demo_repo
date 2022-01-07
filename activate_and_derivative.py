import numpy as np 

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def tanh(Z):
    A = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
    return A

def derivative_sigmoid(A):
    dZ = A(1-A)
    return dZ

def derivative_tanh(A):
    dZ = 1 - np.power(A,2)
    return dZ
