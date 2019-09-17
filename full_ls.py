import numpy as np
import tools
def main(data_nodes,labels,test_nodes,degree):
    '''
    Need to perform full least squares on Ax = b, where x is our coefficients.
    Then evalute the resulting polynomial at the test nodes and return the outputs
    in an array. Good luck.
    '''
    A = create_vandermonde(data_nodes,degree)
    Q, R = np.linalg.qr(A)
    c = tools.solve_qr(Q,R,labels)
    #print('c for full', c)
    approx_values = []
    for node in test_nodes:
        value = 0
        for j in range(len(c)):
            value += c[j]*(node**j)
        approx_values.append(value)

    return approx_values,c
    # plotting_nodes = np.linspace(-5,5,num_nodes)
    # plotting_values = []
    # for node in plotting_nodes:
    #     value = 0
    #     for j in range(len(c)):
    #         value += c[j]*(node**j)
    #     plotting_values.append(value)
    #print(approx_values)
    #return approx_values, c,plotting_values

def create_vandermonde(nodes,degree):
    '''
    Create the vandermonde associated to given nodes up to given degree
    '''
    V = np.array([np.array([x**i for i in range(degree+1)]) for x in nodes])
    return V
# A = [[2,1],[6,1],[20,1],[30,1],[40,1]]
# A = np.array([np.array(row) for row in A])
#
# b = np.array([20,18,10,6,2])
# main(A,b,[0,100])
