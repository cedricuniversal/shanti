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
#download
#os.system('curl https://zenodo.org/record/6532501/files/CESM_EA_SPI.nc?download=1 --output CESM_EA_SPI.nc')

file_name = 'CESM_EA_SPI.nc'
ds = xr.open_dataset(file_name)
spi = ds['spi']
spi = spi.dropna(dim='time')

import pandas as pd
dates = np.load('test_time.npy')
results = []
for d in dates:
    ts = pd.to_datetime(str(d))
    date0 = ts.strftime('%Y-%m-%d')
    delta = pd.Timedelta(days=31*6)
    ts = ts + delta
    date1 = ts.strftime('%Y-%m-%d')
    rec = spi.sel(time=slice(date0, date1))[-2:].values.mean(0)
    results.append(rec)

results = np.array(results)
np.save('test_submission.npy', results)
#test_submission = np.load('test_submission400.npy')
#mse_values = []

#for i in range(120):
#    mse = mean_squared_error(test_submission[i], results[i])
#    mse_values.append(mse)

#print(np.array(mse_values).mean())