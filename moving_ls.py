import numpy as np
import math
import tools
import full_ls
import Vander2d
def main(data_nodes,labels,eval_nodes,degree = 3):
    '''
    Execute moving least squares on the given system Ax = b, centered at each test
    node and return 'all' approx values. Returns matrix where rows are coefficients
    generated for each node.
    '''
    approx_values = []
    node_coefs = []
    for node in eval_nodes:
        node_coef = moving_ls(data_nodes,labels,node,degree)
        node_coefs.append(node_coef)
        value = 0
        for j in range(len(node_coef)):
            value += node_coef[j]*(node**j)
        approx_values.append(value)
    return approx_values, node_coefs


def moving_ls(data_nodes,labels,eval_node,degree,delta = .05, full = False):
    '''
    compute moving least sqaures on given Ax = b system at the given node.
    Return the coefficients of the approximated polynomial
    '''
    if full:
        A = Vander2d.v2d(data_nodes,degree)
        Q,R = np.linalg.qr(A)
        coefficients = tools.solve_qr(Q,R,labels)
        return coefficients
    WA,Wb = create_weighted_system2d(data_nodes,labels,degree,eval_node,delta)
    Q,R= np.linalg.qr(WA)
    coefficients = tools.solve_qr(Q,R,Wb)
    return coefficients

#weight function is e^(|x-node|/delta)
def weight(x,node,delta):
    exponent = -1*(abs(x-node))/delta
    return math.exp(exponent)

def create_weighted_system(nodes,labels,degree,eval_node,delta):
    WA = []
    Wb = []
    for indx, node in enumerate(nodes):
        weight_val = weight(node,eval_node,delta)
        if weight_val > 0:
            WA.append(np.array([weight_val*(node**j) for j in range(degree+1)]))
            Wb.append(weight_val*labels[indx])
    return np.array(WA), np.array(Wb)

'''
2 Dimensions
'''
def weight2d(node, eval_node,delta):
    diff = np.array([node[i] - eval_node[i] for i in range(len(node))])
    alpha = -1*(np.linalg.norm(diff,2))/delta
    return np.exp(alpha)

def build_v2_row(node, degree):
    row = []
    for i in range(degree+1):
        for j in range(degree+1):
            if (i+j) <= degree:
                row.append((node[0]**i)*(node[1]**j))
    return np.array(row)

def create_weighted_system2d(nodes,labels,degree,eval_node,delta):
    WA = []
    Wb = []
    for indx, node in enumerate(nodes):
        weight_val = weight2d(node,eval_node,delta)
        if weight_val > 0:
            row = weight_val*(build_v2_row(node,degree))
            WA.append(row)
            Wb.append(weight_val*labels[indx])
    return np.array(WA), np.array(Wb)
