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

import pyts.metrics

file_name = 'CESM_EA_SPI.nc'
ds = xr.open_dataset(file_name)
spi = ds['spi']
spi = spi.dropna(dim='time')

print(spi.shape)
# test_input (120, 13, 20, 3)

n = 120

lag = 3

nsteps = 3

size = spi.shape[0]

truth = []
test = []

for i in range(n):
    offset = random.randint(0, size - lag - nsteps)
    test.append(spi[offset:offset + lag].values.reshape(13,20,3))
    truth.append(spi[offset+lag+nsteps-1])

print(np.array(truth).shape)
print(np.array(test).shape)
np.save('test_input_sample.npy', test)
np.save('truth_sample.npy', truth)

import numpy as np
import xarray as xr
import pandas
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error

prefix = ''

# cluster_labels = np.load('cluster_labels_train.npy')     # Trained on train

cluster_labels = np.load(prefix + 'cluster_labels_spi.npy')  # Trained on spi
dtw_clusters = cluster_labels.reshape((13, 20))

# models = np.load('models_train.npy', allow_pickle=True)

models = np.load(prefix + 'models_spi.npy', allow_pickle=True)

# predictions_train = np.load('saved_predictions_train.npy')
# truth_train = np.load('saved_truth_train.npy')
# tests_train = np.load('saved_tests_train.npy')
# predictions_train = np.load('saved_predictions_test.npy')
# truth_train = np.load('saved_truth_test.npy')
# tests_train = np.load('saved_tests_test.npy')
# predictions_spi = np.load('saved_predictions_spi.npy')
# truth_spi = np.load('saved_truth_spi.npy')
# tests_spi = np.load('saved_tests_spi.npy')
# LOAD EVALUATION SET

# ..

# test_input (120,13,20,3)
# test_input = np.load('test_input.npy')     # Trained on train

test_input = np.load('test_input_sample.npy')  # Trained on train

print('test_input', test_input.shape)

# print(test_input)

# test_input = test_input.reshape((360,13,20))


# Generate Clustered EVALUATION SET
run = True

if run:
    n_clusters = len(models)
    data = test_input

    n_steps = 3

    # steps = data['time'].to_series()

    lats = list(range(13))  # data['lat']
    lons = list(range(20))  # data['lon']

    df_spi = [pandas.DataFrame() for i in range(n_clusters)]

    columns = []
    points = []
    index = []
    n = 0

    for triplets in data:
        # print(triplets.shape)
        # 13,20,3
        for step in triplets.T:
            # print(step.T.shape)
            stamp = step.T

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
                    points[cluster].append(float(stamp[i][j]))

                    #    float(data.loc[t, float(lat), float(lon)].data))
                    # points[cluster].append(float(data.loc[t, float(lat), float(lon)].data))

            for cluster in range(n_clusters):
                chunk = pandas.DataFrame([points[cluster]], columns=columns[cluster], index=[n])
                df_spi[cluster] = pandas.concat([df_spi[cluster], chunk])

            n += 1

print(n, ' time stamps')
# print(df_spi)

# Predict on Evaluation set using VAR and DTW clusters, recompose
if run:
    n_clusters = len(models)
    n_steps = 3
    lag = 3
    save = True
    saved_predictions_eval = []
    saved_truth_eval = []
    saved_tests_eval = []

    mse_values = []

    n = int(len(df_spi[0]) / 3)

    for offset in range(n):  # len(df_spi[0]) - lag - n_steps):

        predictions = [[] for i in range(n_clusters)]

        for cluster in range(n_clusters):
            test_extract = df_spi[cluster].iloc[offset * 3:offset * 3 + lag].values
            var_model = VAR(test_extract)
            prediction = models[cluster].forecast(var_model.endog, steps=n_steps)
            predictions[cluster] = prediction[-1]

        recomposed_prediction = [[0 for i in range(len(lons))] for j in range(len(lats))]

        # Recompose the predictions
        for cluster in range(n_clusters):
            for tup in range(len(index[cluster])):
                i, j = index[cluster][tup]
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

        # truth = eval[offset -1 + lag + n_steps]

        if save:
            saved_predictions_eval.append(recomposed_prediction)
            # saved_truth_eval.append(truth)
            # saved_tests_eval.append(spi[offset*3:offset*3+lag].values.reshape(3,260))
        # mse = mean_squared_error(recomposed_prediction, truth)
        # mse_values.append(mse)
        # print('DTW_KMEANS_VAR_drought Mean MSE on %i tests: %f'%(len(mse_values), np.array(mse_values[-100:]).mean()))

    # np.save('saved_predictions_eval.npy', np.array(saved_predictions_eval))
    # np.save('saved_truth_eval.npy', np.array(saved_truth_eval))
    # np.save('saved_tests_eval.npy', np.array(saved_tests_eval))

# saved_predictions_eval = np.load('saved_predictions_eval.npy')

print('test_submission sample', np.array(saved_predictions_eval).shape)
np.save('test_submission_sample.npy', np.array(saved_predictions_eval))

truth = np.load('truth_sample.npy')
print(np.array(saved_predictions_eval).shape, np.array(truth).shape)
mse_values = []

for i in range(120):
    mse = mean_squared_error(saved_predictions_eval[i], truth[i])
    mse_values.append(mse)
    # print(mse)
print('mean mse', np.array(mse_values).mean())