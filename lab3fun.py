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
    yOLD = list(y)

    priorOLD = [yOLD.count(x)/float(N) for x in classes]
    priorOLD = np.matrix(priorOLD).reshape((len(classes),1))

    prior = np.zeros((len(classes),1))
    for c in classes:
        numerator = sum(((y == c)*W.T).T)
        denominator = sum(W)
        # denominatorn ska inte behövas om vi vet att sum(W) alltid blir 1...
        prior[c] = numerator/denominator

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

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)
    #print 'init wCur= ', wCur

    for i_iter in range(0, T):
        print 'ITER ', i_iter
        #print 'Wcur =', wCur
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, y, wCur))

        # do classification for each point
        #vote = classifiers[-1].classify(X) # RENAMED
        ht = classifiers[-1].classify(X)
        #print 'ht = ', ht

        # TODO : Fill in the rest, construct the alphas etc.
        # ==========================

        # compute priors w.r.t. current weight wCur:
        pk = computePrior(y, wCur)
        # compute errors w.r.t. current weight wCur (step 2):
        i = ht == y # i is zero if ht != ci
        #print 'i = ', i
        et = sum((wCur.T*(1-i)).T)
        #print 'et =', et
        # compute alphas w.r.t. current weight wCur (step 3):
        alphat = 0.5*(math.log(1-et)-math.log(et))
        # update weights (step 4):
        wCurUN1 = i*wCur.T*math.exp(-alphat) # if ht = ci
        wCurUN2 = (1-i)*wCur.T*math.exp(alphat) # if ht != ci
        wCurUN = np.array([sum(x) for x in zip(wCurUN1, wCurUN2)])
        # normalize:
        wCur = (wCurUN/sum(wCurUN.T)).T
        #print 'wCur= ', wCur

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

        # TODO : implement classificiation when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================

        # ==========================

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)


# The implemented functions can now be summarized another classifer, the `BoostClassifier` class. This class enables boosting different types of classifiers by initializing it with the `base_classifier` argument. No need to add anything here.


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


# ## Run some experiments
#
# Call the `testClassifier` and `plotBoundary` functions for this part.


#testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)

#testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)

#plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris',split=0.7)

# Now repeat the steps with a decision tree classifier.

#testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)

#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)

#testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)

#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)

#plotBoundary(DecisionTreeClassifier(), dataset='iris',split=0.7)

#plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)

# ## Bonus: Visualize faces classified using boosted decision trees
#
# Note that this part of the assignment is completely voluntary!
# First, let's check how a boosted decision tree classifier performs
# on the olivetti data. Note that we need to reduce the dimension a
# bit using PCA, as the original dimension of the image vectors is
# `64 x 64 = 4096` elements.


#testClassifier(BayesClassifier(), dataset='olivetti',split=0.7, dim=20)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='olivetti',split=0.7, dim=20)


# You should get an accuracy around 70%. If you wish, you can compare
# this with using pure decision trees or a boosted bayes classifier.
# Not too bad, now let's try and classify a face as belonging to
# one of 40 persons!


#X,y,pcadim = fetchDataset('olivetti') # fetch the olivetti data
#xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,0.7) # split into training and testing
#pca = decomposition.PCA(n_components=20) # use PCA to reduce the dimension to 20
#pca.fit(xTr) # use training data to fit the transform
#xTrpca = pca.transform(xTr) # apply on training data
#xTepca = pca.transform(xTe) # apply on test data
# use our pre-defined decision tree classifier together with the implemented
# boosting to classify data points in the training data
#classifier = BoostClassifier(DecisionTreeClassifier(), T=10).trainClassifier(xTrpca, yTr)
#yPr = classifier.classify(xTepca)
# choose a test point to visualize
#testind = random.randint(0, xTe.shape[0]-1)
# visualize the test point together with the training points used to train
# the class that the test point was classified to belong to
#visualizeOlivettiVectors(xTr[yTr == yPr[testind],:], xTe[testind,:])
