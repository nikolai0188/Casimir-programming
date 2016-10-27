import numpy
from numpy import linalg
import matplotlib as mp
import scipy
from scipy import misc

#A = misc.imread('face.png')

#A_R = A[1,:,:]
#A_G = A[2,:,:]
#A_B = A[3,:,:]

A = [[4, 6, 2],[12, 5, 8],[3, 5, 14]]

A_SYM = numpy.dot(numpy.transpose(A),A)

D,U = linalg.eig(A_SYM)

D = numpy.dot(D,D)

A_SYM = numpy.dot(A,numpy.transpose(A))

D,V = linalg.eig(A_SYM)

print(A)
print(U)
print(V)
print(D)
