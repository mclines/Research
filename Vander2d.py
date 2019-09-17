import numpy as np

def v2d(nodes, degree):
    A = []
    for node in nodes:
        row = build_v2_row(node,degree)
        A.append(row)
    return np.array(A)


def build_v2_row(node, degree):
    row = []
    for i in range(degree+1):
        for j in range(degree+1):
            if (i+j) <= degree:
                row.append((node[0]**i)*(node[1]**j))
    return np.array(row)
#
# A = v2d([[2,3]],degree = 3)
# tools.print_matrix(A)
