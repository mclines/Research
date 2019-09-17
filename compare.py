import numpy as np
import os
import argparse
import full_ls
import moving_ls
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()
#parser.add_argument('-nodes', default = 1000, type = int, required = False)
parser.add_argument('-degree', default = 3, type = int, required = False)
parser.add_argument('-a',default = -3, type = int, required = False)
parser.add_argument('-b', default = 3, type = int, required = False)
args = parser.parse_args()


def execute(args):
    fig = plt.figure()
    test_nodes, test_values = create_test()
    func = 'abs(x)'
    num_nodes = 10
    degree = 3
    nodes = np.linspace(args.a,args.b ,num_nodes)
    node_values = test_function(nodes)
    print('start full')
    full_approx_values, full_coef= full_ls.main(nodes,node_values,test_nodes, degree)
    print('start moving')
    moving_approx_values, moving_coef = moving_ls.main(nodes,node_values,test_nodes,degree)
    print('end moving')
    #plot_inputs = [full_coef[j]*(x**j) for j in range(len)]
    residual_nodes_m = [abs(test_values[j] - moving_approx_values[j]) for  j in range(len(test_values))]
    residual_nodes_f = [abs(test_values[j] - full_approx_values[j]) for  j in range(len(test_values))]

    ax = fig.add_subplot(221)
    ax.title.set_text(str(num_nodes) + '- ' + func)
    ax.axis([-3,3,-1,2])
    ax.plot(nodes,node_values,nodes,plot_values,'r',test_nodes,moving_approx_values,'bo')

    bx = fig.add_subplot(223)
    bx.title.set_text('Error: Moving')
    bx.plot(test_nodes,residual_nodes_m, 'go')

    cx = fig.add_subplot(224)
    cx.title.set_text('Error: Full')
    cx.plot(test_nodes,residual_nodes_f, 'yo')

    plt.subplots_adjust(bottom=0.1, right=.9, left = 0.12,top=.9, hspace = .6)
    file_name = "./results/ls_{}_d{}_n{}.png".format(func, args.degree, len(nodes))
    plt.savefig(file_name)
    return None

def create_vandermonde(nodes,degree):
    '''
    Create the vandermonde associated to given nodes up to given degree
    '''
    V = np.array([np.array([x**i for i in range(degree+1)]) for x in nodes])
    return V

def create_test():
    '''
    create appropriate b values for system, along with random test nodes with
    test values based on the predefined test function.
    '''
    test_nodes = np.array([np.random.uniform(-3,3) for i in range(1)])
    test_values = np.array([test_function(x) for x in test_nodes])
    return test_nodes, test_values


def test_function(x):
    return abs(x)

def runge(x):
    return 1/(1+25*x**2)

#Run#
execute(args)
