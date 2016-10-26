import numpy as np
from numpy import linalg as LA
import matplotlib as mp
import scipy
from scipy import misc

#A = misc.imread('face.png')

#A_R = A[1,:,:]
#A_G = A[2,:,:]
#A_B = A[3,:,:]

A = [[4, 6, 2],[12, 5, 8],[3, 5, 14]]

A_SYM = np.dot(np.transpose(A),A)

D,U = LA.eig(A_SYM)

D = np.dot(D,D)

A_SYM = np.dot(A,np.transpose(A))

D,V = LA.eig(A_SYM)

print(D)
