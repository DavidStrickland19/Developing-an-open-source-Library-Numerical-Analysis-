'''
##############################################################################################
                                FIXED POINT ITERATION   3.9
#############################################################################################'''                            

#EXERCISE NUMBER 3
#Y0 = 1/2      Y1 = 0.5433216987500 are fixed and take h= .125
# y(n+1) = 1.3333333(Y1) -  .3333333(Y0) - 2h(yn)* ln(yn)
#y0 = Y1
                                
import math

f = lambda x, Y1, Y0, h: (1.3333333 * (Y1)) - ( .3333333 * (Y0)) - ( (2*h*x)) * (math.log(x))                              

Y0 = 0.5
Y1 = 0.5433216987500
h = .125
x0 = Y1
n = 10

#--------------------------------------------------------------------------------------------

def iter3(f,x,Y1,Y0,h,n):
    for i in range(n):
        x= f(x,Y1,Y0, h)
        print ('Iteration *{} is: {}'.format(i,x))
        
iter3(f,x0,Y1,Y0,h,n)




















'''
###############################################################################################
                        FAMILY OF MATRICIES
############################################################################################ '''
                        
                        
import numpy
def HnMatrix(n):
    H = numpy.empty([n, n], dtype=numpy.float64)
    
    for i in range(n):
        for j in range (n):
            H[i,j] = 1.0/((i+1) + (j+1) - 1.0)
    
    return H

def KnMatrix(n):
    K = numpy.empty([n, n], dtype=numpy.float64)
    
    for i in range(n):
        for j in range (n):
            if( i == j ):
                K[i,j] = 2.0
            elif( abs(i - j ) == 1):
                K[i,j] = -1.0
            else:
                K[i,j] = 0.0
                
    return K
                
def TnMatrix(n):
    T = numpy.full([n, n], 0.0)

    for i in range(n):
        for j in range (n):
            if( i == j ):
                T[i,j] = 4.0
            elif( abs(i - j) == 1):
                T[i,j] = 1.0
            else:
                T[i,j] = 0.0
                
    return T

def AnMatrix(n):
    A = numpy.empty([n, n], dtype=numpy.float64)
    
    for i in range(n):
        for j in range (n):
            if( i == j ):
                A[i,j] = 1.0
            elif( i - j == 1):
                A[i,j] = 4.0
            elif( i - j == -1):
                A[i,j] = -4.0
            else:
                A[i,j] = 0.0
                
    return A



'''###########################################################################################

                                 Gauss without Pivoting
#############################################################################################'''
        
        
# Gaussian Elimination Without Partial Pivoting




import numpy as np

def GENP(A, b):
    
    #Gaussian elimination with no pivoting.
    #input: A is an n x n nonsingular matrix
    #       b is an n x 1 vector
    # output: x is the solution of Ax=b.
    # post-condition: A and b have been modified. 
    
    n =  len(A)
    if b.size != n:
        raise ValueError("Invalid argument: incompatible sizes between A & b.", b.size, n)
    for pivot_row in range(n-1):
        for row in range(pivot_row+1, n):
            multiplier = A[row][pivot_row]/A[pivot_row][pivot_row]
            #the only one in this column since the rest are zero
            A[row][pivot_row] = multiplier
            for col in range(pivot_row + 1, n):
                A[row][col] = A[row][col] - multiplier*A[pivot_row][col]
            #Equation solution column
            b[row] = b[row] - multiplier*b[pivot_row]
    print(A)
    print(b)
    x = np.zeros(n)
    k = n-1
    x[k] = b[k]/A[k,k]
    while k >= 0:
        x[k] = (b[k] - np.dot(A[k,k+1:],x[k+1:]))/A[k,k]
        k = k-1
    return x

#caller
if __name__ == "__main__":
 
#paramter
    A = np.array[()]
    b =  np.array[()]
    
    #caller
    print(GENP(np.copy(A), np.copy(b)))




##############################################################################################




''' #########################################################################################
                                GAUSSIAN ELIMINATION WITH PIVOTING                      
-------------------------------------------------------------------------------------- '''

def algorithm_74(Amatrix, bvector):
    (numrows, numcols) = Amatrix.shape
    a = Amatrix.copy()
    b = bvector.copy()
    
    #making sure that our matrix is square
    if(numrows == numcols):
        
        for i in range(numrows):
            #the elimination steps to get the upper triangular form, looking for pivots
            
            am = abs(a[i,i])
            pivot = i
        
            #looping through to make sure pivot index is the index with the largest abs value
            for j in range(i+1, numrows):
                if( abs(a[j,i]) > am):
                    am = abs(a[j,i])
                    pivot = j
                
                if( pivot > i ):
                    #execute row interchange so that the pivot is in the ith row
                    for k in range(i, numcols):
                        temp = numpy.copy(a[i,k])
                        a[i,k] = a[pivot,k]
                        a[pivot,k] = temp

                
                    hold = numpy.copy(b[i,0])
                    b[i,0] = b[pivot,0]
                    b[pivot,0] = hold
            
            for j in range(i+1, numrows):
                m = a[j,i]/a[i,i]
            
                for k in range(i+1, numrows):
                    a[j,k] = a[j,k] - m*a[i,k]
            
                b[j,0] = b[j,0] - m*b[i,0]
            
            
        #back solver part
        x = numpy.matrix(numpy.zeros((numrows,1), dtype = numpy.float64))
        n = numrows - 1
        x[n,0] = b[n,0]/a[n,n]    
        
        #going up along the diagonal
        for i in range(n-1,-1,-1):
            sum = 0
            for j in range(i+1, numrows):
                sum = sum + a[i,j]*x[j,0]
            x[i,0] = (b[i,0] - sum)/a[i,i]
    
    return x 


















'''
##########################################################################################
                         LU FACTORIZATION AND BACK SOLVE (7.4)                          
########################################################################################## 
'''





def LU_factors(Amat):
    a = Amat.copy()
    
    (n, numcol) = a.shape
    
    new_order = [i for i in range(n)]
    
    if(n == numcol):
        #Factor. w/ Pivoting
        for i in range(n):
            #pivoting section
            am = abs(a[i,i])
            p = i
            for j in range(i+1, n):
                if(abs(a[j,i]) > am):
                    am = abs(a[j,i])
                    p = j
            #if the pivot index is greater than the current i index (i loops columnwise)
            #then interchange Row=pivot with Row=i
            if(p > i):
                for k in range(n):
                    temp = numpy.copy(a[i,k])
                    a[i,k] = a[p,k]
                    a[p,k] = temp
                    
                #track the swapped indices
                temp = new_order[i]
                new_order[i] = new_order[p]
                new_order[p] = temp
            
            #the pivoting is now complete
            #now begin the elimination step
            for j in range(i+1, n):
                a[j,i] = a[j,i]/a[i,i]
                for k in range(i+1, n):
                    a[j,k] = a[j,k] - a[j,i]*a[i,k]
                    
        #the pivoting is now complete so,: compute L and U
        U = numpy.matrix(numpy.zeros((n,n), dtype = numpy.float64))
        L = numpy.matrix(numpy.zeros((n,n), dtype = numpy.float64))
        
        for i in range(n):
            U[i,i] = a[i,i]
            L[i,i] = 1.0
            for j in range(n):
                if( i < j ):
                    U[i,j] = a[i,j]
                elif( i > j ):
                    L[i,j] = a[i,j]
        
                
    return L, U
#------------------------------------------------------------------------------------------------

def LU_forward_backward(L,U, bvec):
    l = L.copy()
    u = U.copy()
    b = bvec.copy()
    
    (n, numcol) = l.shape

    x = numpy.matrix(numpy.zeros((n,1), dtype = numpy.float64))
    
    if(n == numcol):
        #Solve Ly = b
        y = numpy.matrix(numpy.zeros((n,1), dtype = numpy.float64))
        y[0,0] = b[0,0]
        for i in range(1, n):
            sum = 0.0
            for j in range(0, i):
                 sum = sum + l[i,j]*y[j,0]
            y[i,0] = b[i,0] - sum
         
        #Solve Ux = y
        x[n-1, 0] = y[n-1, 0]/u[n-1,n-1]
        for i in range(n-2, -1, -1):
            sum = 0.0
            for j in range(i+1, n):
                sum = sum + u[i,j]*x[j, 0]
            x[i,0] = (y[i,0] - sum)/u[i,i]
        
    return x



'''
###############################################################################################
'''
'''                     NORMS AND CONDITION NUMBERS 7.5                                     '''
''' ####################################################################################### '''




'''
Input: vector
Returns: infinity norm, i.e., maximum absolute value entry
'''
def vecInfNorm(v):
    return numpy.max(numpy.abs(v))
        
'''
Input: matrix
Returns: infinity matrix norm, i.e., maximum row sum
'''
def matrixInfNorm(A):
    a = numpy.copy(A)
    a = abs(a)
    rowsums = numpy.sum(a, axis=1).tolist()
    max_s = rowsums[0]
    for elem in rowsums:
        if elem > max_s:
            max_s = elem
    return max_s

'''
Input: vector
Returns: Euclidean 2-norm
'''
def vecEuclidnorm(v, p):
    return math.pow(numpy.sum(numpy.pow(numpy.abs(v), 2.0)), 0.5)

'''
Input: vector, p
Returns: vector P norm
'''
def vecPnorm(v, p):
    return math.pow(numpy.sum(numpy.pow(numpy.abs(v), p)), 1.0/p)

'''
Input: n x n matrix
output: matrix 2 norm
'''
def matrix2norm(Amat):

    A = numpy.matmul(Amat, numpy.transpose(Amat))
    (eigvalues, eigenvectors) = numpy.linalg.eig(A)
    mxe = max(eigvalues)
    return math.sqrt(mxe)

'''
 The Condition Number Estimator Algorithm
''' 
def algorithm_79(A):
    #solve the system Ayi = yi+1, i.e., LUyi = yi+1 as a way of solving A-inverse(A^-1) yi+1 = yi
    '''
    a = A.copy()
    (n, numcol) = A.shape
    alpha = matrixInfNorm(A)
    (X,idx)=LU_factors(a)
    #y = numpy.random.rand(numrows, magnitude)
    y = numpy.random.rand(n, 1)
    for i in range(5):
        y = y/vecInfNorm(y)
        y = algorithm_78(L, U, y)
    '''   
    a = A.copy()
    (n, numcol) = A.shape
    alpha = matrixInfNorm(A)
    (X,idx)=LU_factors(a)
    #y = numpy.random.rand(numrows, magnitude)
    y = numpy.random.rand(n, 1)
    for i in range(5):
        print(y)
        y = y/vecInfNorm(y)
        y = LU_forward_backward(X, idx, y)
        
        
    
    v = vecInfNorm(y)
    print("v inf")
    print(v)
    print("alpha")
    print(alpha)
    print(y)
    print(alpha * v)
    
    
'''##########################################################################################'''
'''##########################################################################################'''