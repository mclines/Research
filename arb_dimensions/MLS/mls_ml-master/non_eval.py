import numpy as np
import csv
import mls_arb_dimension as mls

def approximate_nons(data_nodes,labels,num_nons,degree = 10):
    '''
    Generate the near optimal nodes along with their approximated labels.

    Inputs: Data nodes as an array of nodes and each node itself is an array (even if just a scalar)

            Labels are stored in an as array where each label is in its own array (even scalars)
            this allows for arbitrary dimension in the processing.

            num_nons is the number of near optimal nodes desired.

    Output: An array containg the near optimal nodes and an array containing the associated labels
    '''
    nons = generate_nodes(num_nons)
    print( 'Line 19')
    non_labels, _ = mls.main(data_nodes, labels, nons, degree = 10)
    return nons, non_labels


def generate_nodes(num_nons):
    '''
    For general purposes, will choose chebyshev nodes or fekete points
    '''
    chebyshev_nodes = []
    for k in range(num_nons):
        node = np.cos((2*k*np.pi)/(2*num_nons))
        chebyshev_nodes.append(np.array([node]))
    return np.array(chebyshev_nodes)


def read_in_csv(file_name):
    with open(file_name,'r') as f:
        array = list(csv.reader(f, delimiter=','))
        array = np.array(array, dtype = np.float)
        labels = np.array([np.array([label]) for label in array[:,-1]])
        data = array[:,:-1]
        # print(data)
        # print(labels)
        return data,labels

def write_to_csv(file_name, data):
    with open(file_name,'w+') as f:
        write_ = csv.writer(f, delimiter=',')
        for row in data:
            write_.writerow(row)
    return None

def simple_func(x):
    return np.sin(10*x)*(0.8*x)

def ex_gf_func(x):
        sum_ = sum([np.sin(np.pi *(k**2)*(x))/(np.pi*(k**2)) for k in range(1,100)])
        return sum_

def gen_data(num_nodes, interval = [-1,1], func = simple_func):
    a,b = interval[0], interval[1]
    space = np.linspace(a,b,num_nodes)
    data = np.array([np.array([node]) for node in space])
    labels = np.array([func(node) for node in data])
    data_matrix = np.array([np.concatenate((data[j],labels[j])) for j in range(len(data))])
    return data_matrix


def main(file_name,num_nons,degree_approx = 10):
    '''
    Input a csv file name (including the .csv), the number of near optimal nodes
    desired and an optional argument for desired degree of approximation.
    '''
    print(file_name, ' \ \ ', 'Line 72')
    data,labels = read_in_csv(file_name)
    print(file_name, ' \ \ ', 'Line 74')
    nons, non_labels = approximate_nons(data,labels,num_nons,degree = degree_approx)
    print(file_name, ' \ \ ', 'Line 76')
    data_matrix = np.array([np.concatenate((nons[j],non_labels[j])) for j in range(len(nons))])
    fn = file_name[:-4]
    non_file_name = '{}_nons.csv'.format(fn)
    print(file_name, ' \ \ ', 'Line 81')
    write_to_csv(non_file_name,data_matrix)
    return non_file_name

# read_in_csv('test.csv')
#write_to_csv('./generated_data/simple_data.csv',gen_data(10000,func = simple_func))
#main('simple_data.csv',10)
#ead_in_csv('test.csv')
