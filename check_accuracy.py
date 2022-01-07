import numpy as np

def check_result(A,Y):
    result = np.reshape(A-Y, (-1))
    correct = 0
    for i in range(len(result)):
        if result[i]>0 and result[i]<0.5:
            correct = correct+1
        elif result[i]<0 and result[i]>-0.5:
            correct = correct+1
    accuracy = correct/len(Y)
    return(accuracy)



