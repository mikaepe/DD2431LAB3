# Assignment 1

from sklearn.datasets.samples_generator import make_blobs
import numpy as np


def genBlobs(n_samples=20,centers=5,n_features=2):
    #                  200 originally
    # Supplied from the lab material
    X, y = make_blobs(n_samples=n_samples,\
            centers=centers, n_features=n_features,\
            random_state=0)
    return X,y

def mlParams(X,y):
    # Assignment 1, computes ML-estimates of mu_k and Sigma_k    
    # X is data points, y is labels for points

    classes = np.unique(y)             # result: [0,1,2,3,4]

    # compute the mu vectors:
    m = np.zeros((len(classes),2))
    for j,c in enumerate(classes):
        i = y == c                      # return True/False length of y 
        xlc = X[i,:]                    # the x with class c
        m[j,:] = sum(xlc)/sum(i)        # store mean in mu

    # compute the Sigma matrix: (TODO)

    S = 1
    return m, S

X,y = genBlobs()

#print X
#print y

mlParams(X,y)

