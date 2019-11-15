import numpy as np
import itertools
import tools
import time
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

def main(data_nodes,labels,eval_nodes,degree, delta = 0.05, method = 'QR'):
    '''
    Execute moving least squares on the given system Ax = b, centered at each test
    node and return 'all' approx values. Returns matrix where rows are coefficients
    generated for each node.
    '''
    approx_values = []
    node_coefs = []
    for indx, node in enumerate(eval_nodes):
        print(indx, ' out of ', len(eval_nodes))
        node_coef = moving_ls(data_nodes,labels,node,degree,delta)
        node_coefs.append(node_coef)
        value = eval_coefficients(node_coef,node,degree)
        approx_values.append(value)
    return approx_values, node_coefs

def moving_ls(data_nodes,labels,eval_node,degree,delta = .05, method = 'QR'):
    '''
    compute moving least sqaures on given Ax = b system at the given node.
    Return the coefficients of the approximated polynomial
    '''
    t0 = time.time()
    print('MLS: Line 30')
    WA,Wb = create_weighted_system(data_nodes,labels,degree,eval_node,delta)
    t1 = time.time()
    #print('Time to create_weighted_system:',t1-t0)
    coefficients = tools.solve_system(WA,Wb,method)
    return coefficients

def weight(node, eval_node,norm_val,delta):
    alpha = -1*(norm_val)/delta
    return np.exp(alpha)

def create_arb_row(dimension, degree):
    '''
    create the degree combinations for arb. node to create vandermonde row,
    or to evaluate the coefficients of arb. degree polynomial in arb. dimension.
    '''
    set_up = [[i for i in range(degree+1)] for j in range(dimension)]
    comb = list(itertools.product(*set_up))
    n = len(comb)
    degree_combination = [comb[i] for i in range(n) if sum(comb[i]) <= degree]
    return degree_combination


def build_row(node,degree_comb):
    '''
    Take in a node along with degree combinations, and create appropriate vandermonde
    row.
    '''
    row = [1]*len(degree_comb)
    for dim, degree in enumerate(degree_comb):
        for indx, deg in enumerate(degree):
            row[dim]*= (node[indx]**deg)
    return np.array(row)


def eval_coefficients(coef,node,degree):
    '''
    Evaluate the polynomial with given coefficients at the given node.
    '''
    powers = create_arb_row(len(node),degree)
    value = 0
    if len(powers) > len(coef):
        raise ValueError("Not enought coefficients:{}".format(len(coef)))
    for indx, power in enumerate(powers):
        raised = np.power(node,power)
        val = coef[indx]*np.product(raised)
        value += val
    return value

def create_weighted_system(nodes,labels,degree,eval_node,delta):
    WA = []
    Wb = []
    beta = 20.7232658369*delta #distance
    degree_comb = create_arb_row(len(nodes[0]),degree)
    for indx, node in enumerate(nodes):
        norm_val = np.linalg.norm((node-eval_node),2)
        #print('norm_val',norm_val)
        if norm_val < beta:
            #print('here')
            weight_val = weight(node,eval_node,norm_val, delta)
            row = weight_val*(build_row(node,degree_comb))
            WA.append(row)
            Wb.append(weight_val*labels[indx])
    return np.array(WA), np.array(Wb)
