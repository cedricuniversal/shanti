from svr import svr
import numpy as np
import xarray as xr

test_input = np.load('test_input_sample.npy')
#test_input = np.load('test_input.npy')

test_input = test_input.reshape(360, 13, 20)

cluster_labels = np.load('cluster_labels_spi.npy')
cluster_labels = cluster_labels.reshape((13, 20))

test0 = np.array([z[cluster_labels==0] for z in test_input])
#print(test0.shape)
test1 = np.array([z[cluster_labels==1] for z in test_input])

file_name = 'CESM_EA_SPI.nc'
ds = xr.open_dataset(file_name)
spi = ds['spi']
nt, nlat, nlon = spi.shape
spi = spi[5:] # remove NaNs
spi = spi.data

#print('# of NaNs: ', len(spi[spi != spi]))

train0 = np.array([z[cluster_labels==0] for z in spi])
train1 = np.array([z[cluster_labels==1] for z in spi])

pred0 = svr(train0, test0)
pred1 = svr(train1, test1)

foo = np.zeros((120, 13, 20))
for i in range(120):
    foo[i][cluster_labels==0] = pred0[i]
    foo[i][cluster_labels==1] = pred1[i]

#np.save('test_submission.npy', foo)
np.save('test_submission_sample_cluster.npy', foo)
