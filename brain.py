from initialize_parameters import initialize
from forward_propagation import forward
from cost_function import cost_function
from back_propagation import back_propagation
from activate_and_derivative import derivative_tanh, sigmoid, tanh
from layer_size import layer_size
from update_parameters import update_parameters

def brain(X, Y, iteration, learning_rate):
    n_x, n_h, n_y = layer_size(X,Y)

    parameters = initialize(n_x, n_h, n_y)

    for _ in range(iteration):
        A2, cache = forward(X, parameters)
        
        print(cost_function(A2,Y))
        
        grads = back_propagation(parameters, cache, X, Y)      
        
        parameters = update_parameters(parameters=parameters, grads=grads, learning_rate=learning_rate)  

    return parameters



    
