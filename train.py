
import os
os.system('pip install pyts')

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
import pyts.metrics

#download
os.system('curl https://zenodo.org/record/6532501/files/CESM_EA_SPI.nc?download=1 --output CESM_EA_SPI.nc')

file_name = 'CESM_EA_SPI.nc'
ds = xr.open_dataset(file_name)
spi = ds['spi']
spi = spi.dropna(dim='time')

# Calculate DTW matrix

train_size = 1200

#data = train[-train_size:]
data = spi[-train_size:]              # remove comment
#data = spi.interpolate_na(dim='time')
#data = data.fillna(0)

lats = spi['lat'].to_series().values

lons = spi['lon'].to_series().values

dtw_sim_spi = [[[[0 for j in range(len(spi['lon']))] for i in range(len(spi['lat']))]
            for j in range(len(spi['lon']))] for i in range(len(spi['lat']))]

for i in range(len(spi['lat'])):
    for j in range(len(spi['lon'])):
        for k in range(len(spi['lat'])):
            for l in range(len(spi['lon'])):
                dtw_sim_spi[i][j][k][l] = pyts.metrics.dtw(data[:, i, j], data[:, k, l], method='fast')


np.save('dtw_sim_spi.npy', dtw_sim_spi)

# Aggregate the dtw mean per square on SPI

import numpy as np
import matplotlib.pyplot as plt

dtw_sim_array = np.array(dtw_sim_spi)
dtw_mean = dtw_sim_array.mean((0,1))

#plt.imshow(dtw_mean, interpolation='none')
#plt.show()

# Find best clustering

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

tuples = []
X = []

for i in range(len(lats)):
  for j in range(len(lons)):
    lat = lats[i]
    lon = lons[j]
    tuples.append([lat, lon])
    X.append([dtw_mean[i][j]])


n_clusters_best = 0
silhouette_best = 0

for i in range(2,30):
  n_clusters=i
  clusterer = KMeans(n_clusters=n_clusters, random_state=10)
  cluster_labels = clusterer.fit_predict(X)
  silhouette_avg = silhouette_score(X, cluster_labels)
  if silhouette_avg > silhouette_best:
    silhouette_best = silhouette_avg
    n_clusters_best = n_clusters
    #plt.imshow(cluster_labels.reshape((13,20)), interpolation='none')
    #plt.title("%i clusters silhouette : %f"%(n_clusters,silhouette_avg))
    #plt.show()

n_clusters = n_clusters_best
clusterer = KMeans(n_clusters=n_clusters, random_state=10)
cluster_labels = clusterer.fit_predict(X)
dtw_clusters = cluster_labels.reshape((13,20))
np.save('cluster_labels_spi.npy', cluster_labels)

print("DTW cLuster labels stored")