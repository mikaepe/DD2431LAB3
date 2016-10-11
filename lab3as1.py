# Assignment 1

import labfuns as fun
import numpy as np
import math as m

def mlParams(X,y):
    # Assignment 1, computes ML-estimates of mu_k and Sigma_k    
    # X is data points, y is labels for points

    classes = np.unique(y)              # array with unique classes
    nFeat = np.shape(X)[1]              # number of features in data
    mm = np.zeros((len(classes),nFeat))  
    S = np.zeros((len(classes),nFeat,nFeat))
    for k,c in enumerate(classes):
        i = y == c                      # return True/False length of y 
        Xk = X[i,:]                     # the X with class c
        Nk = sum(i)                     # no of data points class c
        mu = sum(Xk)/Nk                 # store mean in mu
        mm[k,:] = mu                     # store mu in m-matrix
        xic = Xk - mu                   # center the data for S-comput.
        #S[k,:,:] = xic.reshape(nFeat,1)*xic/Nk
        S[k,:,:] = np.diag(sum(xic*xic))/Nk # Naive (S(m,n) = 0, n != m)
    
    return mm, S

X,y = fun.genBlobs(200,5,2)
#                  200/5/2 orig
'''
print 'X = ',X
print 'y = ',y
'''
mu,S = mlParams(X,y)
print 'mu = ',mu
print 'Sigma = ',S


'''
fun.plotGaussian(X,y,mu,S)


def computePrior(y):
    N = len(y)
    classes = list(np.unique(y))
    y = list(y)
    Pk = [y.count(x)/float(N) for x in classes]
    return Pk

Pk = computePrior(y)
    
def classifyBayes(X,prior,mu,S):
    
    detDiag = np.prod(np.diag(S))
    D1 = -0.5*m.log(detDiag)

    SinvDiag = 1.0/S





classifyBayes(X,Pk[0],mu[0,:],S[0,:,:])
'''    
