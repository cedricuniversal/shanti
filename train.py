
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
    plt.imshow(cluster_labels.reshape((13,20)), interpolation='none')
    plt.title("%i clusters silhouette : %f"%(n_clusters,silhouette_avg))
    plt.show()

n_clusters = n_clusters_best
clusterer = KMeans(n_clusters=n_clusters, random_state=10)
cluster_labels = clusterer.fit_predict(X)
dtw_clusters = cluster_labels.reshape((13,20))
np.save('cluster_labels_spi.npy', cluster_labels)

# Generate Clustered SPI SET

if True:

    data = spi

    n_steps = 3

    steps = data['time'].to_series()

    lats = data['lat']
    lons = data['lon']

    df_spi = [pd.DataFrame() for i in range(n_clusters)]

    columns = []
    points = []
    index = []

    for t in steps:
        columns = [[] for i in range(n_clusters)]
        points = [[] for i in range(n_clusters)]
        index = [[] for i in range(n_clusters)]

        for i in range(len(lats)):
            for j in range(len(lons)):
                lat = lats[i]
                lon = lons[j]

                cluster = dtw_clusters[i][j]

                columns[cluster].append((float(lat), float(lon)))

                index[cluster].append((i, j))

                points[cluster].append(float(data.loc[t, float(lat), float(lon)].data))

        for cluster in range(n_clusters):
            app = pd.DataFrame([points[cluster]], columns=columns[cluster], index=[t])
            df_spi[cluster] = pd.concat([df_spi[cluster], app])

# TRAIN clustered models ON SPI SET, OUTPUT MODELS[n_clusters]

if True:
    lag = 3

    models = [[] for i in range(n_clusters)]
    for cluster in range(n_clusters):
        var_model = VAR(df_spi[cluster].values)
        model_fit = var_model.fit(lag)
        models[cluster] = model_fit

np.save('models_spi.npy', np.array(models))

# Predict on SPI set using VAR and DTW clusters, recompose

if True:
    n_steps = 3
    lag = 3

    save = True
    saved_predictions_spi = []
    saved_truth_spi = []
    saved_tests_spi = []

    mse_values = []
    for offset in range(0, len(df_spi[0]) - lag - n_steps):  # SPI EDIT

        predictions = [[] for i in range(n_clusters)]

        for cluster in range(n_clusters):
            test_extract = df_spi[cluster].iloc[offset:offset + lag].values  # SPI EDIT
            var_model = VAR(test_extract)
            prediction = models[cluster].forecast(var_model.endog, steps=n_steps)
            # print(prediction.shape)
            predictions[cluster] = prediction[-1]

        recomposed_prediction = [[0 for i in range(len(lons))] for j in range(len(lats))]

        # Recompose the predictions
        for cluster in range(n_clusters):
            for tup in range(len(index[cluster])):
                i, j = index[cluster][tup]
                # print(i,j,cluster,tup)
                recomposed_prediction[i][j] = predictions[cluster][tup]

        # plt.imshow(recomposed_prediction)
        # plt.colorbar()
        # plt.clim(-3, 3);
        # plt.title("DTW %i cluster VAR prediction"%n_clusters)
        # plt.show()

        # plt.imshow(test[offset -1 + 8 + n_steps])
        # plt.colorbar()
        # plt.clim(-3, 3);
        # plt.title('Gound truth')
        # plt.show()

        truth = spi[offset - 1 + lag + n_steps]

        if save:
            saved_predictions_spi.append(recomposed_prediction)
            saved_truth_spi.append(truth)
            saved_tests_spi.append(spi[offset:offset + lag].values.reshape(3, 260))
        mse = mean_squared_error(recomposed_prediction, truth)
        mse_values.append(mse)
    print('DTW_KMEANS_VAR_drought Mean MSE on %i tests: %f' % (len(mse_values), np.array(mse_values).mean()))

np.save('saved_predictions_spi.npy', np.array(saved_predictions_spi))
np.save('saved_truth_spi.npy', np.array(saved_truth_spi))
np.save('saved_tests_spi.npy', np.array(saved_tests_spi))