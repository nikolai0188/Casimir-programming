import numpy as np
from numpy import linalg as LA
from numpy import random
import matplotlib as mp
import matplotlib.image as mpimg
import scipy
from scipy import linalg as LA2
from scipy import misc
import matplotlib.pyplot as plt
#%matplotlib inline

img = mpimg.imread('face.png')
B = np.zeros(img.shape)

for i in range(3):

    A = img[:,:,i]
    
    #A_SYM = numpy.dot(numpy.transpose(A),A)

    #D,U = linalg.eig(A_SYM)

    #D = numpy.dot(D,D)

    #A_SYM = numpy.dot(A,numpy.transpose(A))

    #D,V = linalg.eig(A_SYM)

    #rows = 10
    #columns = 6
    #A = random.randint(0,256,size=(rows,columns))
    #print(A)

    compression = 600

    U,s,V = LA2.svd(A,full_matrices=False)

    s=s[:-compression]

    S = np.diag(s)

    U = np.delete(U,np.s_[U.shape[1]-compression:],1)
    V = np.delete(V,np.s_[V.shape[0]-compression:],0)

    B[:,:,i] = np.dot(U,np.dot(S,V))

plt.imshow(B)
