import os, sys, pdb
import netCDF4 as nc
import xarray as xr
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
#from thundersvm import SVR
from sklearn.model_selection import StratifiedShuffleSplit, KFold,  GridSearchCV
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

def svr(train, test):
    train = train.reshape(len(train), -1)
    nhist = 3
    lead = 3
    trainskip = 1
    npcs = 10 # 25 explains 90%
    pca = PCA(random_state=0, n_components=npcs, copy=False, whiten=True)
    pca.fit(train)
    pcs = pca.transform(train)
    xvr = pca.explained_variance_ratio_
    ntot = len(pcs)
    tlast = ntot - lead
    indx = np.arange(ntot)
    tmp = [(pcs[n - nhist : n], pcs[n + lead -1]) for n in range(nhist, tlast+1)]
    X, Y = [np.array(z).reshape(len(z), -1) for z in list(zip(*tmp))]
    tmp = [(train[n - nhist : n], train[n + lead -1]) for n in range(nhist, tlast+1)]
    Xp, Yp = [np.array(z).reshape(len(z), -1) for z in list(zip(*tmp))]
    # begin non-random train-test split
    ntrain = int(1.1*len(X))
    xtrain, ytrain, xtest, ytest = X[:ntrain], Y[:ntrain], X[ntrain:], Y[ntrain:]
    ytestphys = Yp[ntrain:]
    train_ix = np.arange(ntrain)
    test_ix = np.arange(ntrain, len(X))
    # end non-random train-test split
    xtrain, ytrain = [z[::trainskip] for z in [xtrain, ytrain]]
    ntrain = len(xtrain)
    ntest = len(xtest)
    print([z.shape for z in [xtrain, ytrain, xtest, ytest]])
    xtest = pca.transform(test.reshape(len(test), -1)).reshape(120, -1)
    regrs = []
    pred = []
    for ipc in range(npcs):
        regr = SVR()
        regr.fit(xtrain, ytrain[:,ipc])
        pred1 = regr.predict(xtest)
        regrs.append(regr)
        pred.append(pred1)
    pred = np.array(pred).T
    predphys = pca.inverse_transform(pred)#.reshape(120, 13, 20)
    return predphys