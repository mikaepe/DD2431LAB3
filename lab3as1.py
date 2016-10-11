# Assignment 1

import labfuns as fun
import numpy as np

def mlParams(X,y):
    # Assignment 1, computes ML-estimates of mu_k and Sigma_k    
    # X is data points, y is labels for points

    classes = np.unique(y)             # result: [0,1,2,3,4]
    nFeat = np.shape(X)[1]

    # compute the mu vectors:
    m = np.zeros((len(classes),nFeat))
    S = np.zeros((len(classes),nFeat,nFeat))
    for k,c in enumerate(classes):
        i = y == c                      # return True/False length of y 
        xlc = X[i,:]                    # the x with class c
        Nk = sum(i)
        mu = sum(xlc)/Nk                # store mean in mu
        m[k,:] = mu
        xic = xlc - mu
        #S[k,:,:] = xic.reshape(nFeat,1)*xic/Nk
        S[k,:,:] = np.diag(sum(xic*xic))/Nk
    
    return m, S

X,y = fun.genBlobs(200,5,2)
#                  200/5/2 orig
print X
print 'y = ', y

m,S = mlParams(X,y)
print m
print S

fun.plotGaussian(X,y,m,S)

