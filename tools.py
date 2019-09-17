import numpy as np

def print_matrix(A):
  for i in range(len(A)):
    for j in range(len(A[0])):
        if A[i][j] == 0:
            A[i][j] = 0
        print("%.2f" % A[i][j], end= " ")
    print("\n")
  return None


def is_valid_matrix(A):
   if not isinstance(A,list) or (len(A) == 0):
       return False
   if not isinstance(A[0],list) or (len(A) == 0):
       return False
   len_vector = len(A[0])
   print("length vector = ", len_vector)
   for vector in A:
       print("Length of this vector",vector,len(vector))
       if not (isinstance(vector,list)) or len(vector) != len_vector:
              return False
       for elem in vector:
           if not isinstance(elem,float) and not isinstance(elem,int):
               print(isinstance(elem,int))
               return False
   return True


def forward_sub(A,b):
   n = len(A)
   x = [0]*n
   for i in range(n):
       num = b[i] - sum([A[i][j]*x[j] for j in range(i)])
       x[i] = num/A[i][i]
   return x

def backward_sub(A,b):
   n = len(A)
   x = [0]*n
   for i in range(n-1,-1,-1):
       num = b[i] - sum([A[i][j]*x[j] for j in range(i+1,n)])
       x[i] = num/A[i][i]
   return x

def solve_qr(Q,R,b):
    QT = np.transpose(Q)
    QTb = np.matmul(QT,b)
    c = backward_sub(R,QTb)
    return c
