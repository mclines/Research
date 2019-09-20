import numpy as np
import math
import tools
import full_ls
def main(data_nodes,labels,eval_nodes,degree = 3, full = False):
    '''
    Execute moving least squares on the given system Ax = b, centered at each test
    node and return 'all' approx values. Returns matrix where rows are coefficients
    generated for each node.
    '''
    approx_values = []
    node_coefs = []
    for node in eval_nodes:
        node_coef = moving_ls(data_nodes,labels,node,degree,full)
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
        A = full_ls.create_vandermonde(data_nodes,degree)
        Q,R = np.linalg.qr(A)
        coefficients = tools.solve_qr(Q,R,labels)
        return coefficients
    WA,Wb = create_weighted_system(data_nodes,labels,degree,eval_node,delta)
    Q,R= np.linalg.qr(WA)
    coefficients = tools.solve_qr(Q,R,Wb)
    return coefficients

#weight function is e^(|x-node|/delta)
def weight(x,node,norm_val,delta):
    exponent = -1*(norm_val)/delta
    return math.exp(exponent)

def create_weighted_system(nodes,labels,degree,eval_node,delta):
    WA = []
    Wb = []
    beta = 20.7232658369*delta
    for indx, node in enumerate(nodes):
        norm_val = abs((node-eval_node))
        if norm_val < beta:
            weight_val = weight(node,eval_node,norm_val, delta)
            WA.append(np.array([weight_val*(node**j) for j in range(degree+1)]))
            Wb.append(weight_val*labels[indx])
    return np.array(WA), np.array(Wb)
