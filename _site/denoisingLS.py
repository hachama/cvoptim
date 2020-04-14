import numpy as np
import cv2 as cv
from scipy.linalg import toeplitz
import scipy
import scipy.sparse.linalg as spla


#
lam = 1

# Load an color image in grayscale
img  = cv.imread('input.jpg',0)
w, h = img.shape
img  = cv.imread('input.jpg',0)
imgn = img + np.random.normal(0, 5,img.shape)

N = w*h

col = np.zeros((h),  dtype=None, order='C')
col[0]=-1
row = np.zeros((h),  dtype=None, order='C')
row[0]=-1
row[1]=1
I = toeplitz(col,row)
I[h-1, h-2]=1

y = imgn.flatten()

# a=[ [1, -1, 0], [0, 1, 0] , [0, 0, 1] ]
A = scipy.sparse.kron(np.eye(w,dtype=int),I)
AA = scipy.sparse.identity(N) + lam *A.transpose().dot(A)
x = spla.spsolve(AA,y) 

X =  x.reshape(w,h)
print(img.shape, X.shape)
cv.namedWindow('Test LS',cv.WINDOW_NORMAL)
cv.imshow('Test LS',np.concatenate( (img, X.astype(np.uint8)), axis=1))
cv.waitKey()
cv.destroyAllWindows()

