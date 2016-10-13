#!/usr/bin/python
# coding: utf-8

# # Lab 3: Bayes Classifier and Boosting

from fun import *

# ------- CALCULATIONS // FUNCTION CALLS -------

# PRINT THINGS OR NOT
print_debug = True

as1 = 0
if as1:
    X,y = genBlobs(centers=5)
    mu,S = mlParams(X,y)
    plotGaussian(X,y,mu,S)

as3 = 0
if as3:
    # Call the `testClassifier` and `plotBoundary` functions for this part.

    testClassifier(BayesClassifier(), dataset='iris', split=0.7)
    testClassifier(BayesClassifier(), dataset='vowel', split=0.7)
    plotBoundary(BayesClassifier(), dataset='iris',split=0.7)
    #classifiers, alphas = fun.trainBoost(fun.BayesClassifier(), X, y, T=2)


as5 = 1
if as5:

    testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)
    testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)
    plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris',split=0.7)

# Now repeat the steps with a decision tree classifier.

as6 = 0
if as6:


    testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)
    testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)
    testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)
    testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)
    plotBoundary(DecisionTreeClassifier(), dataset='iris',split=0.7)
    plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)



ff = 0
if ff:

    # ## Bonus: Visualize faces classified using boosted decision trees
    #
    # Note that this part of the assignment is completely voluntary!
    # First, let's check how a boosted decision tree classifier performs
    # on the olivetti data. Note that we need to reduce the dimension a
    # bit using PCA, as the original dimension of the image vectors is
    # `64 x 64 = 4096` elements.


    testClassifier(BayesClassifier(), dataset='olivetti',split=0.7, dim=20)



    testClassifier(BoostClassifier(DecisionTreeClassifier(),\
            T=10), dataset='olivetti',split=0.7, dim=20)


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
