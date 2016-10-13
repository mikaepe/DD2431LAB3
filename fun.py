from __future__ import absolute_import, division, print_function
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from matplotlib.colors import ColorConverter
import random as rnd
from sklearn.datasets.samples_generator import make_blobs
from sklearn import decomposition, tree

# import seaborn as sns
# sns.set()


# # Lab 3: Bayes Classifier and Boosting

import numpy as np
from scipy import misc
from imp import reload
#from labfuns import *
import random, math


#######################################################################
### THE ADJUSTED FUNCTIONS mlParams(), computePriors(), etc. ##########
#######################################################################


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
        #denominatorn ska inte behovas om vi vet att sum(W) alltid blir 1...
        #prior[c] = numerator/denominator
        prior[c] = sum(((y == c)*W.T).T)

    # TODO snygga till det har och ta bort lite joks

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



#######################################################################
### ADDITIONAL SUPPLIED FUNCTIONS #####################################
#######################################################################


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ellip.set_alpha(0.25)

    ax.add_artist(ellip)
    return ellip


# Splits data into training and test set, pcSplit defines the percent of
# the data should be used as training data.
def trteSplit(X,y,pcSplit,seed=None):
    # Compute split indices
    Ndata = X.shape[0]
    Ntr = int(np.rint(Ndata*pcSplit))
    Nte = Ndata-Ntr
    np.random.seed(seed)    
    idx = np.random.permutation(Ndata)
    trIdx = idx[:Ntr]
    teIdx = idx[Ntr:]
    # Split data
    xTr = X[trIdx,:]
    yTr = y[trIdx]
    xTe = X[teIdx,:]
    yTe = y[teIdx]
    return xTr,yTr,xTe,yTe,trIdx,teIdx


# Splits data into training and test set, pcSplit defines the percent of
# the data should be used as training data. The major difference to
# trteSplit is that we select the percent from each class individually.
# This means that we are assured to have enough points for each class.
def trteSplitEven(X,y,pcSplit,seed=None):
    labels = np.unique(y)
    xTr = np.zeros((0,X.shape[1]))
    xTe = np.zeros((0,X.shape[1]))
    yTe = np.zeros((0,),dtype=int)
    yTr = np.zeros((0,),dtype=int)
    trIdx = np.zeros((0,),dtype=int)
    teIdx = np.zeros((0,),dtype=int)
    np.random.seed(seed)
    for label in labels:
        classIdx = np.where(y==label)[0]
        NPerClass = len(classIdx)
        Ntr = int(np.rint(NPerClass*pcSplit))
        idx = np.random.permutation(NPerClass)
        trClIdx = classIdx[idx[:Ntr]]
        teClIdx = classIdx[idx[Ntr:]]
        trIdx = np.hstack((trIdx,trClIdx))
        teIdx = np.hstack((teIdx,teClIdx))
        # Split data
        xTr = np.vstack((xTr,X[trClIdx,:]))
        yTr = np.hstack((yTr,y[trClIdx]))
        xTe = np.vstack((xTe,X[teClIdx,:]))
        yTe = np.hstack((yTe,y[teClIdx]))

    return xTr,yTr,xTe,yTe,trIdx,teIdx


def fetchDataset(dataset='iris'):
    if dataset == 'iris':
        X = genfromtxt('irisX.txt', delimiter=',')
        y = genfromtxt('irisY.txt', delimiter=',',dtype=np.int)-1
        pcadim = 2
    elif dataset == 'wine':
        X = genfromtxt('wineX.txt', delimiter=',')
        y = genfromtxt('wineY.txt', delimiter=',',dtype=np.int)-1
        pcadim = 0
    elif dataset == 'olivetti':
        X = genfromtxt('olivettifacesX.txt', delimiter=',')
        X = X/255
        y = genfromtxt('olivettifacesY.txt', delimiter=',',dtype=np.int)
        pcadim = 20
    elif dataset == 'vowel':
        X = genfromtxt('vowelX.txt', delimiter=',')
        y = genfromtxt('vowelY.txt', delimiter=',',dtype=np.int)
        pcadim = 0
    else:
        print("Please specify a dataset!")
        X = np.zeros(0)
        y = np.zeros(0)
        pcadim = 0

    return X,y,pcadim


def genBlobs(n_samples=200,centers=5,n_features=2):
    X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features,random_state=0)
    return X,y


# Scatter plots the two first dimension of the given data matrix X
# and colors the points by the labels.
def scatter2D(X,y):
    labels = np.unique(y)
    Ncolors = len(labels)
    xx = np.arange(Ncolors)
    ys = [i+xx+(i*xx)**2 for i in range(Ncolors)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    c = 1.0
    for label in labels:
        classIdx = np.where(y==label)[0]
        Xclass = X[classIdx,:]
        plt.scatter(Xclass[:,0],Xclass[:,1],linewidths=1,\
                s=25,color=colors[label],marker='o',alpha=0.75)
        c += 1.

    plt.show()


def plotGaussian(X,y,mu,sigma):
    labels = np.unique(y)
    Ncolors = len(labels)
    xx = np.arange(Ncolors)
    ys = [i+xx+(i*xx)**2 for i in range(Ncolors)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    c = 1.0
    for label in labels:
        classIdx = y==label
        Xclass = X[classIdx,:]
        plot_cov_ellipse(sigma[label], mu[label])
        plt.scatter(Xclass[:,0],Xclass[:,1],linewidths=1,\
                s=25,color=colors[label],marker='o',alpha=0.75)
        c += 1.

    plt.show()


# The function below, `testClassifier`, will be used to try out the different datasets.
# `fetchDataset` can be provided with any of the dataset arguments `wine`, 
# `iris`, `olivetti` and `vowel`.
# Observe that we split the data into a **training** and a **testing** set.
def testClassifier(classifier, dataset='iris', dim=0, split=0.7, ntrials=100):

    X,y,pcadim = fetchDataset(dataset)

    means = np.zeros(ntrials,);

    for trial in range(ntrials):

        xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,split,trial)

        # Do PCA replace default value if user provides it
        if dim > 0:
            pcadim = dim

        if pcadim > 0:
            pca = decomposition.PCA(n_components=pcadim)
            pca.fit(xTr)
            xTr = pca.transform(xTr)
            xTe = pca.transform(xTe)

        # Train
        trained_classifier = classifier.trainClassifier(xTr, yTr)
        # Predict
        yPr = trained_classifier.classify(xTe)

        # Compute classification error
        if trial % 10 == 0:
            print("Trial:",trial,"Accuracy","%.3g" % (100*np.mean((yPr==yTe).astype(float))) )

        means[trial] = 100*np.mean((yPr==yTe).astype(float))

    print("Final mean classification accuracy ",\
            "%.3g" % (np.mean(means)), "with standard deviation", "%.3g" % (np.std(means)))


# ## Plotting the decision boundary
#
# This is some code that you can use for plotting the decision boundary
# boundary in the last part of the lab.
def plotBoundary(classifier, dataset='iris', split=0.7):

    X,y,pcadim = fetchDataset(dataset)
    xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,split,1)
    classes = np.unique(y)

    pca = decomposition.PCA(n_components=2)
    pca.fit(xTr)

    xTr = pca.transform(xTr)
    xTe = pca.transform(xTe)

    pX = np.vstack((xTr, xTe))
    py = np.hstack((yTr, yTe))

    # Train
    trained_classifier = classifier.trainClassifier(xTr, yTr)

    xRange = np.arange(np.min(pX[:,0]),np.max(pX[:,0]),np.abs(np.max(pX[:,0])-np.min(pX[:,0]))/100.0)
    yRange = np.arange(np.min(pX[:,1]),np.max(pX[:,1]),np.abs(np.max(pX[:,1])-np.min(pX[:,1]))/100.0)

    grid = np.zeros((yRange.size, xRange.size))

    for (xi, xx) in enumerate(xRange):
        for (yi, yy) in enumerate(yRange):
            # Predict
            grid[yi,xi] = trained_classifier.classify(np.array([[xx, yy]]))

    
    ys = [i+xx+(i*xx)**2 for i in range(len(classes))]
    colormap = cm.rainbow(np.linspace(0, 1, len(ys)))

    fig = plt.figure()
    # plt.hold(True)
    conv = ColorConverter()
    for (color, c) in zip(colormap, classes):
        try:
            CS = plt.contour(xRange,yRange,\
                    (grid==c).astype(float),15,linewidths=0.25,\
                    colors=conv.to_rgba_array(color))
        except ValueError:
            pass
        trClIdx = np.where(y[trIdx] == c)[0]
        teClIdx = np.where(y[teIdx] == c)[0]
        plt.scatter(xTr[trClIdx,0],xTr[trClIdx,1],\
                marker='o',c=color,s=40,alpha=0.5,\
                label="Class "+str(c)+" Train")
        plt.scatter(xTe[teClIdx,0],xTe[teClIdx,1],\
                marker='*',c=color,s=50,alpha=0.8,\
                label="Class "+str(c)+" Test")
    plt.legend(bbox_to_anchor=(1., 1), loc=2, borderaxespad=0.)
    fig.subplots_adjust(right=0.7)
    plt.show()


def visualizeOlivettiVectors(xTr, Xte):
    N = xTr.shape[0]
    Xte = Xte.reshape(64, 64).transpose()
    plt.subplot(1, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title("Test image")
    plt.imshow(Xte, cmap=plt.get_cmap('gray'))
    for i in range(0, N):
        plt.subplot(N, 2, 2+2*i)
        plt.xticks([])
        plt.yticks([])
        plt.title("Matched class training image %i" % (i+1))
        X = xTr[i, :].reshape(64, 64).transpose()
        plt.imshow(X, cmap=plt.get_cmap('gray'))
    plt.show()


class DecisionTreeClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = DecisionTreeClassifier()
        rtn.classifier = tree.DecisionTreeClassifier(max_depth=Xtr.shape[1]/2+1)
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)
