#!/usr/bin/python
# coding: utf-8

# # Lab 3: Bayes Classifier and Boosting

import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random, math


# Bayes classifier functions to implement
# The lab descriptions state what each function should do.


# NOTE: you do not need to handle the W argument for this part!
# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(y, W=None):
    N = y.shape[0]
    if W is None:
        W = np.ones((N,1))/N
    else:
        assert(W.shape[0] == N)
    classes = list(np.unique(y))
    #Nclasses = np.size(classes)
    #prior = np.zeros((Nclasses,1))
    y = list(y)
    prior = [y.count(x)/float(N) for x in classes]
    prior = np.matrix(prior).reshape((len(classes),1))
    # TODO ska det vara en Cx1-vektor??? Annars ta bort reshape

    return prior


# NOTE: you do not need to handle the W argument for this part!
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
        i = y == c              # return True/False length of labels
        Xk = X[i,:]             # the X with class c
        Nk = sum(i)             # no of data pts class c
        muk = sum(Xk)/Nk        # store mean in muk
        mu[k,:] = muk           # store muk in mu-matrix
        xic = Xk - muk          # center data for S-computation
        S[k,:,:] = np.diag(sum(xic*xic))/Nk     # Naive, (S(m,n) = 0, n != m)

    #print 'mu = ',mu
    #print 'S = ',S
    return mu, S

# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(X, prior, mu, S):

    N = X.shape[0]                          # no of points
    Ncl,Ndim = np.shape(mu)                 # no of classes and features
    logProb = np.zeros((Ncl, N))


    # ==========================
    # TODO : fill in the code to compute the log posterior logProb!

    for k in range(Ncl):
        Sk = S[k,:,:]
        #x = X[0,:] 
        mmu = mu[k,:]
        priorK = prior[k,:]

        SdetDiag = np.prod(np.diag(Sk))
        SinvDiag = np.diag(1.0/np.diag(Sk))     # to avoid division by zero of diag elem.
        
        d1 = -0.5*math.log(SdetDiag)
        #d22 = -0.5*(x-mmu).dot(SinvDiag).dot((x-mmu).T)
        d2 = -0.5*sum(np.multiply((X-mmu).dot(SinvDiag),(X-mmu)).T)
        d3 = math.log(priorK)

        #print 'd1', d1
        #print 'd2', d2
        #print 'd22 ', d22
        #print 'd3', d3

        logProb[k,:] = d2+d1+d3
        # ==========================

        # one possible way of finding max a-posteriori once
        # you have computed the log posterior
    h = np.argmax(logProb,axis=0)
    return h

'''
X,y = genBlobs(10,3,2)
#           200/5/2 originally

mu,S = mlParams(X,y)
print 'mu = ',mu
print 'Sigma = ',S

Pk = computePrior(y)
print 'Priors = ',Pk


classifyBayes(X,Pk,mu,S)
'''




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


# ## Test the Maximum Likelihood estimates
#
# Call `genBlobs` and `plotGaussian` to verify your estimates.

'''
X,y = genBlobs(centers=3)
mu,S = mlParams(X,y)
plotGaussian(X,y,mu,S)
'''


# Call the `testClassifier` and `plotBoundary` functions for this part.


#testClassifier(BayesClassifier(), dataset='iris', split=0.7, ntrials=10)
testClassifier(BayesClassifier(), dataset='iris', split=0.7)

#testClassifier(BayesClassifier(), dataset='vowel', split=0.7)

plotBoundary(BayesClassifier(), dataset='iris',split=0.7)


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

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, y, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)

        # TODO : Fill in the rest, construct the alphas etc.
        # ==========================

        # alphas.append(alpha) # you will need to append the new alpha
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
