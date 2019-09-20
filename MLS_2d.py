import numpy as np
import math
import tools
import full_ls
import Vander2d
def main(data_nodes,labels,eval_nodes,degree, delta = 0.05, full = False):
    '''
    Execute moving least squares on the given system Ax = b, centered at each test
    node and return 'all' approx values. Returns matrix where rows are coefficients
    generated for each node.
    '''
    update_file_main = open('./progress_update_main.txt','a')
    update_file_main.write('Created Data. Reached mls_2d.main() method. Preparing to preform mls.')
    update_file_main.close()
    print('Created Data. Reached mls_2d.main() method. Preparing to preform mls.')
    approx_values = []
    node_coefs = []
    for node in eval_nodes:
        node_coef = moving_ls(data_nodes,labels,node,degree,delta, full)
        node_coefs.append(node_coef)
        print('Evaluating the polynomial at the eval node.')
        value = eval_coefficients(node_coef,node,degree)
        update_file_eval = open('./progress_update_eval.txt','a')
        update_file_eval.write('Evaluated the polynomial successfully.')
        update_file_eval.close()
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
    update_file_cws = open('./progress_update_create_system.txt','a')
    update_file_cws.write('Created weighted system successfully. Preparing to compute the QR decomp.')
    update_file_cws.close()
    print('Created weighted system successfully. Preparing to compute the QR decomp.')
    Q,R= np.linalg.qr(WA)
    update_file_qr = open('./progress_update_qr.txt','a')
    update_file_qr.write('Preformed QR successfully. Preparing to evalute the polynomial at eval node.')
    update_file_qr.close()
    print('Preformed QR successfully. Preparing to evalute the polynomial at eval node.')

    coefficients = tools.solve_qr(Q,R,Wb)
    return coefficients

def weight2d(node, eval_node,norm_val,delta):
    alpha = -1*(norm_val)/delta
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
    beta = 20.7232658369*delta
    for indx, node in enumerate(nodes):
        norm_val = np.linalg.norm((node-eval_node),2)
        if norm_val < beta:
            weight_val = weight2d(node,eval_node,norm_val, delta)
            row = weight_val*(build_v2_row(node,degree))
            WA.append(row)
            Wb.append(weight_val*labels[indx])
    return np.array(WA), np.array(Wb)



def eval_coefficients(coefficients,node,degree):
    value = 0
    counter = 0
    print(len(coefficients))
    for i in range(degree+1):
        for j in range(degree+1):
            if (i+j) <= degree:
                value += coefficients[counter]*((node[0]**i)*(node[1]**j))
                counter += 1
    return value
