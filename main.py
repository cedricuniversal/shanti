import netCDF4 as nc
import xarray as xr
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
import random

from random_sampling import random_sample
from svr import svr
from dtw_cluster_svr import dtw_cluster_svr
from cluster_svr import cluster_svr

train, test, truth = random_sample(120)

print(train.shape, test.shape, truth.shape)

#predictions_dtw = dtw_cluster_svr(train, test, npcs=25)

cluster_labels = np.load('cluster_labels_spi.npy')
predictions_dtw = cluster_svr(train, test, cluster_labels)
print('Predictions', predictions_dtw.reshape(-1).shape, truth.shape)
print('DTW ', mean_squared_error(predictions_dtw.reshape(-1), truth.reshape(-1)))

#test = test.reshape(len(test)*lead_time, 13, 20)
test = test.reshape((len(test), 13*20, 3))
predictions_svr = svr(train, test)
print('SVR ', mean_squared_error(predictions_svr.reshape(-1), truth.reshape(-1)))

svr_error = np.array(predictions_svr.reshape(predictions_svr.shape[0], 13, 20)-truth).mean(0)
plt.imshow(svr_error)#.reshape((13, 20)))
plt.title('MEAN Error SVR')
plt.show()

dtw_error = np.array(predictions_dtw-truth).mean(0)
plt.imshow(np.array(dtw_error))#.reshape((13, 20)))
plt.title('MEAN Error DTW')
plt.show()


