import numpy as np

def check_result(A,Y):
    """
    This function will calculate the accuracy of the machine
    Argument: 
    A -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Return:
    accuracy -- correct_predict/number_of_example
    """
    result = np.reshape(A-Y, (-1))
    correct = 0
    for i in range(len(result)):
        if result[i]>0 and result[i]<0.5:
            correct = correct+1
        elif result[i]<0 and result[i]>-0.5:
            correct = correct+1
    accuracy = correct/len(Y)
    return(accuracy)



