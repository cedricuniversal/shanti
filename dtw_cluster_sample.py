from svr_cluster import svr
import numpy as np
import xarray as xr

def dtw_cluster_svr(test_input):
    test_input = test_input.reshape(360, 13, 20)

    cluster_labels = np.load('cluster_labels_spi.npy')
    n_clusters = len(set(cluster_labels))
    cluster_labels = cluster_labels.reshape((13, 20))

    file_name = 'CESM_EA_SPI.nc'
    ds = xr.open_dataset(file_name)
    spi = ds['spi']
    nt, nlat, nlon = spi.shape
    spi = spi[5:] # remove NaNs
    spi = spi.data

    predictions = [[] for i in range(n_clusters)]

    for cluster in range(n_clusters):

        test = np.array([z[cluster_labels == cluster] for z in test_input])

        train = np.array([z[cluster_labels == cluster] for z in spi])

        predictions[cluster] = svr(train, test)

    foo = np.zeros((120, 13, 20))

    for cluster in range(n_clusters):
        for i in range(120):
            foo[i][cluster_labels == cluster] = predictions[cluster][i]

    return foo


if __name__ == "__main__":

    test_input = np.load('test_input_sample.npy')

    results = dtw_cluster_svr(test_input)

    np.save('test_submission_sample.npy', results)

