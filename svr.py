import os, sys, pdb
import netCDF4 as nc
import xarray as xr
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedShuffleSplit, KFold,  GridSearchCV
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

def svr(train, test):
    #test = test.reshape(len(test) * lead_time, 13, 20)
    C = 0.01
    gamma = 0.0001
    print('Train, test', train.shape, test.shape)
    train = train.reshape(len(train), -1)
    nhist = 3
    lead = 3
    #trainskip = 1
    #xvr = pca.explained_variance_ratio_
    #ntot = len(pcs)
    print(test.shape)
    print(train.shape)
    ntot = train.shape[0]
    #pcs = train
    tlast = ntot - lead
    indx = np.arange(ntot)

    #tmp = [(pcs[n - nhist : n], pcs[n + lead -1]) for n in range(nhist, tlast+1)]
    #X, Y = [np.array(z).reshape(len(z), -1) for z in list(zip(*tmp))]

    #tmp = [(pcs[n - nhist : n], pcs[n + lead -1]) for n in range(nhist, tlast+1)]
    #Xt, Y = [np.array(z).reshape(len(z), -1) for z in list(zip(*tmp))]

    tmp = [(train[n - nhist : n], train[n + lead -1]) for n in range(nhist, tlast+1)]
    Xp, Yp = [np.array(z).reshape(len(z), -1) for z in list(zip(*tmp))]

    # begin non-random train-test split

    ntrain = int(1.1*len(Xp))
    xtrain, ytrain= Xp[:ntrain], Yp[:ntrain]

    #xtest, ytest =  X[ntrain:], Y[ntrain:]

    ytestphys = Yp[ntrain:]
    train_ix = np.arange(ntrain)
    test_ix = np.arange(ntrain, len(Xp))
    # end non-random train-test split
    #xtrain, ytrain = [z[::trainskip] for z in [xtrain, ytrain]]
    ntrain = len(xtrain)

    npcs = 10 # 25 explains 90%
    pca = PCA(random_state=0, n_components=npcs, copy=False, whiten=True)

    print('xtrain', xtrain.shape)
    pca.fit(train.reshape(len(train), -1))

    #train_pca = pca.transform(train)
    #test_pca = pca.transform(test)

    # print([z.shape for z in [xtrain, ytrain, xtest, ytest]])
    print(test.shape)
    #xtest = pca.transform(test.reshape(test.shape[0]*test.shape[2], test.shape[1]))#.reshape(len(test), -1)
    #xtrain = pca.transform(xtrain.reshape(len(train), -1))  # .reshape(len(test), -1)

    xtest = test.reshape(test.shape[0], test.shape[1]*test.shape[2]) # .reshape(-1, test.shape[1]*3)

    ntest = len(xtest)
    regrs = []
    pred = []

    cluster_size = train.shape[1]
    for target in range(cluster_size):
        regr = SVR() #kernel='rbf', C=C, gamma=gamma)
        regr.fit(xtrain, ytrain[:, target])
        #print(xtest.shape)
        pred1 = regr.predict(xtest)
        regrs.append(regr)
        pred.append(pred1)
    pred = np.array(pred).T
    #predphys = pca.inverse_transform(pred)#.reshape(120, 13, 20)
    return pred