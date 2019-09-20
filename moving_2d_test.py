import numpy as np
import math
import tools
import time
import MLS_2d_eval
import mls_2d
import traceback

def speed_test():
    eval_nodes = np.array([ np.array([np.random.randn(1)[0],np.random.randn(1)[0]]) ]) #np.random.randn(1)
    interval = [-1,1]
    interval_str = str(interval)
    file1 = open('./text_results/Results-Test1.txt','a')
    for degree in [10]:
        for num_nodes in [100]:
            print('Creating data')
            space = np.linspace(interval[0],interval[1],num_nodes)
            data = MLS_2d_eval.create_data(space)
            labels = np.array([f(node) for node in data])
            for flag in [False, True]:
                output_statement = 'For Degree = {}, Num Nodes = {}^2:, eval_node = {}, interval:[{}],Full:{}'.format(degree,num_nodes,str(eval_nodes[0]),interval_str,str(flag))
                print('------------------------------------------------------------------')
                print(output_statement)
                times = []
                errors = []
                for round in range(1):
                    t0 = time.time()
                    try:
                        approx_value, coef = mls_2d.main(data, labels, eval_nodes, degree, full = flag)
                    except Exception as ex:
                        error_file = open('./progess_error.txt','a')
                        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                        message = template.format(type(ex).__name__, ex.args)
                        error_file.write(message)
                        error_file.write(traceback.format_exc())
                        error_file.close()
                        print(traceback.format_exc())
                        print(message)
                        return None
                    t1 = time.time()
                    eval_time = t1-t0
                    actual_value = f(eval_nodes[0])
                    error = abs(actual_value - approx_value[0])
                    times.append(eval_time)
                    errors.append(error)
                average_time = sum(times)/len(times)
                max_time = max(times)
                #average_error = sum(errors)/len(errors)
                max_error = max(errors)
                #min_error = min(errors)
                print('Average Time: ',average_time)
                print('Max time: ', max_time)
                #print('Average Error: ',average_error)
                print('Max Error: ', max_error)
                #print('Min Error: ', min_error)

                str_avg_time = 'average_time: {}\n'.format(average_time)
                str_break = '------------------------------------------------------------------\n'
                str_max_time = 'max_time: {}\n'.format(max_time)
                str_max_error = 'max_error: {}\n'.format(max_error)
                file1.write(output_statement+'\n')
                file1.write(str_break)
                file1.write(str_avg_time)
                file1.write(str_max_time)
                file1.write(str_max_error)
                file1.write(str_break)
                file1.write('\n \n \n')
    file1.close()
    return None

def f(z):
    return np.sin(np.sqrt(z[0]**2+z[1]**2))+np.cos(z[0])+np.exp(z[1])

def runge(x):
    return 1/(1+25*x**2)


speed_test()
