# Assignment 1

import labfuns as fun
import numpy as np

def mlParams(X,y):
    # Assignment 1, computes ML-estimates of mu_k and Sigma_k    
    # X is data points, y is labels for points

    classes = np.unique(y)              # array with unique classes
    nFeat = np.shape(X)[1]              # number of features in data
    m = np.zeros((len(classes),nFeat))  
    S = np.zeros((len(classes),nFeat,nFeat))
    for k,c in enumerate(classes):
        i = y == c                      # return True/False length of y 
        Xk = X[i,:]                     # the X with class c
        Nk = sum(i)                     # no of data points class c
        mu = sum(Xk)/Nk                 # store mean in mu
        m[k,:] = mu                     # store mu in m-matrix
        xic = Xk - mu                   # center the data for S-comput.
        #S[k,:,:] = xic.reshape(nFeat,1)*xic/Nk
        S[k,:,:] = np.diag(sum(xic*xic))/Nk # Naive (S(m,n) = 0, n != m)
    
    return m, S

X,y = fun.genBlobs(200,5,2)
#                  200/5/2 orig
print X
print 'y = ', y

m,S = mlParams(X,y)
print m
print S

fun.plotGaussian(X,y,m,S)

