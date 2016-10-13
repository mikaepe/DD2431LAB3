#!/usr/bin/python
# coding: utf-8

# # Lab 3: Bayes Classifier and Boosting

import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random, math

# ------- DEFINITIONS OF FUNCTIONS -------

# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, y, W=None):
    assert(X.shape[0]==y.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(y)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    S = np.zeros((Nclasses,Ndims,Ndims))

    for k,c in enumerate(classes):
        i = y == c                  # return True/False length of labels
        Xk = X[i,:]                 # the X with class c
        NkW = sum((i*W.T).T)        # ADDED

        Wk = W[i,:]                 # ADDED
        XkW = np.multiply(Wk,Xk)    # ADDED

        mukW = sum(XkW)/NkW
        xicW = Xk - mukW            # ADDED

        S[k,:,:] = np.diag(sum(Wk*xicW*xicW))/NkW  # ADDED Naive, (S(m,n) = 0, n != m)
        mu[k,:] = mukW

    return mu, S

# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(y, W=None):
    N = y.shape[0]

    if W is None:
        W = np.ones((N,1))/N
    else:
        assert(W.shape[0] == N)

    classes = list(np.unique(y))
    yW = y*W.T
    #yOLD = list(y)

    #priorOLD = [yOLD.count(x)/float(N) for x in classes]
    #priorOLD = np.matrix(priorOLD).reshape((len(classes),1))

    prior = np.zeros((len(classes),1))
    for c in classes:
        #numerator = sum(((y == c)*W.T).T)
        #denominator = sum(W)
        #denominatorn ska inte behövas om vi vet att sum(W) alltid blir 1...
        #prior[c] = numerator/denominator
        prior[c] = sum(((y == c)*W.T).T)

    # TODO snygga till det här och ta bort lite joks

    return prior

# Bayes classifier functions to implement
# The lab descriptions state what each function should do.

# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(X, prior, mu, S):

    N = X.shape[0]                          # no of points
    Ncl,Ndim = np.shape(mu)                 # no of classes and features
    logProb = np.zeros((Ncl, N))

    for k in range(Ncl):
        Sk = S[k,:,:]
        mmu = mu[k,:]

        priorK = prior[k,:]

        SdetDiag = np.prod(np.diag(Sk))
        SinvDiag = np.diag(1.0/np.diag(Sk))     # to avoid division by zero of diag elem.

        d1 = -0.5*math.log(SdetDiag)
        d2 = -0.5*sum(np.multiply((X-mmu).dot(SinvDiag),(X-mmu)).T)
        d3 = math.log(priorK)

        logProb[k,:] = d2+d1+d3

    h = np.argmax(logProb,axis=0)
    return h


# The implementd functions can now be summarized into the `BayesClassifier`
# class, which we will use later to test the classifier,
# no need to add anything else here:
# NOTE: no need to touch this
class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, y, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(y, W)
        rtn.mu, rtn.S = mlParams(X, y, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.S)


# ## Boosting functions to implement
#
# The lab descriptions state what each function should do.


# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, y, T=10):
    # these will come in handy later on
    Npts,Ndims = np.shape(X)

    classifiers = []    # append new classifiers to this list
    alphas = []         # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)
    #print 'init wCur= ', wCur

    for i_iter in range(T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, y, wCur))

        # do classification for each point
        #vote = classifiers[-1].classify(X) # RENAMED
        ht = classifiers[-1].classify(X)

        # TODO : Fill in the rest, construct the alphas etc.
        # ==========================

        # compute priors w.r.t. current weight wCur:
        pk = computePrior(y, wCur)
        # compute errors w.r.t. current weight wCur (step 2):
        i = ht == y # i is zero if ht != ci
        et = sum((wCur.T*(1-i)).T)
        et = max(et,1e-9)           # TODO RIKTIGT FULHAXX!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # compute alphas w.r.t. current weight wCur (step 3):
        alphat = 0.5*(math.log(1-et)-math.log(et))
        # update weights (step 4):
        wCurUN1 = i*wCur.T*math.exp(-alphat) # if ht = ci
        wCurUN2 = (1-i)*wCur.T*math.exp(alphat) # if ht != ci
        wCurUN = np.array([sum(x) for x in zip(wCurUN1, wCurUN2)])
        # normalize:
        wCur = (wCurUN/sum(wCurUN.T)).T
        alphas.append(alphat) # you will need to append the new alpha
        # ==========================

    return classifiers, alphas

# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))

        for c in range(Nclasses):
            suM = np.zeros((Npts,1))
            for t in range(Ncomps):
                ht = classifiers[t].classify(X)
                delta = ht == c*np.ones((1,Npts))
                suM += alphas[t]*delta.reshape((Npts,1))
            votes[:,c] = suM.reshape((1,Npts))

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)


# The implemented functions can now be summarized another classifer, 
# the `BoostClassifier` class. This class enables boosting different 
# types of classifiers by initializing it with the `base_classifier` 
# argument. No need to add anything here.

# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, y):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(y))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, y, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)

