import numpy as np
import math
import tools
import time
import moving_ls
import full_ls

def speed_test():
    eval_node = np.array([0.70453117])#np.random.randn(1)
    for degree in [100]:
        for num_nodes in [10,100,1000,10000]:
            nodes = np.linspace(-2,2,num_nodes)
            labels = runge(nodes)
            times = []
            errors = []
            for round in range(100):
                t0 = time.time()
                approx_value, coef = moving_ls.main(nodes, labels, eval_node, degree)
                t1 = time.time()
                eval_time = t1-t0
                actual_value = runge(eval_node)[0]
                error = abs(actual_value - approx_value[0])
                times.append(eval_time)
                errors.append(error)
            average_time = sum(times)/len(times)
            average_error = sum(errors)/len(errors)
            max_error = max(errors)
            min_error = min(errors)
            output_statement = 'For Degree = {}, Num Nodes = {}, Eval Node = {}:'.format(degree,num_nodes,eval_node)
            print('------------------------------------------------------------------')
            print(output_statement)
            print('For number of nodes = ',num_nodes, 'degree')
            print('Average Time: ',average_time)
            print('Average Error: ',average_error)
            print('Max Error: ', max_error)
            print('Min Error: ', min_error)
    return None


def runge(x):
    return 1/(1+25*x**2)


speed_test()
