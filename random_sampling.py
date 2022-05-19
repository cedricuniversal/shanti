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
def random_sample(n=120):

    file_name = 'CESM_EA_SPI.nc'
    ds = xr.open_dataset(file_name)
    spi = ds['spi']
    spi = spi.dropna(dim='time')

    # print(spi.shape)
    # test_input (120, 13, 20, 3)

    lag = 3

    nsteps = 3

    size = spi.shape[0]

    truth = []
    test = []

    for i in range(n):
        offset = random.randint(0, size - lag - nsteps)
        test.append(spi[offset:offset + lag].values.reshape(13,20,3))
        truth.append(spi[offset+lag+nsteps-1])

    #print(np.array(truth).shape)
    #print(np.array(test).shape)
    np.save('test_input_sample.npy', test)
    np.save('truth_sample.npy', truth)
    return spi.values, np.array(test), np.array(truth)

if __name__ == "__main__":
    train, test, truth = random_sample(120)