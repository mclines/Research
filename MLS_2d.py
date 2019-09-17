import moving_ls
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

def main():
    space = np.linspace(-1,1,10000)
    degree = 11
    data = create_data(space)
    X = np.array([data[i][0] for i in range(len(data))])
    Y = np.array([data[i][1] for i in range(len(data))])
    labels = f(X,Y)
    eval_node = np.array([np.random.rand(1)[0],np.random.rand(1)[0]])
    t0= time.time()
    coefficients = moving_ls.moving_ls(data,labels,eval_node,degree,delta = .05)
    t1 = time.time()
    print(t1-t0)
    eval_value = eval_coefficients(coefficients,eval_node,degree)
    print(abs(eval_value - f(eval_node[0],eval_node[1])))
    #Plot#
    '''
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X,Y,labels,'gray')
    ax.view_init(35, 50)
    plt.savefig('./test 3d')
    '''
    return None



'''
Helpers
'''
def eval_coefficients(coefficients,node,degree):
    value = 0
    counter = 0
    for i in range(degree+1):
        for j in range(degree+1):
            if (i+j) <= degree:
                value += coefficients[counter]*((node[0]**i)*(node[1]**j))
                counter += 1
    return value



def create_data(space):
    data = []
    for x in space:
        for y in space:
            data.append(np.array([x,y]))
    return np.array(data)


def f(x,y):
    return np.sin(np.sqrt(x**2+y**2))
main()
