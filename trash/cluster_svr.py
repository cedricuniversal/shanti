from svrpca import svrpca
import numpy as np
import xarray as xr

def cluster_svr(train, test_input, cluster_labels):
    lead_time = 3
    #test_input = test_input.reshape(len(test_input)*lead_time, 13, 20)

    n_clusters = len(set(cluster_labels))
    cluster_labels = cluster_labels.reshape((13, 20))

    #file_name = 'CESM_EA_SPI.nc'
    #ds = xr.open_dataset(file_name)
    #spi = ds['spi']
    #nt, nlat, nlon = spi.shape
    #spi = spi[5:] # remove NaNs
    #spi = spi.data

    predictions = [[] for i in range(n_clusters)]

    for cluster in range(n_clusters):
        print('Cluster ', cluster)
        cluster_test = np.array([z[cluster_labels == cluster] for z in test_input])
        print(train.shape, cluster_labels.shape)
        print('train shape ', train[0][cluster_labels == cluster].shape)
        cluster_train = np.array([z[cluster_labels == cluster] for z in train])
        #train = train.reshape(len(train) * lead_time, 13, 20)
        print('train test', cluster_train.shape,cluster_test.shape)
        predictions[cluster] = svrpca(cluster_train, cluster_test)


    foo = np.zeros((len(test_input), 13, 20))

    for cluster in range(n_clusters):
        for i in range(len(test_input)):
            foo[i][cluster_labels == cluster] = predictions[cluster][i]
    return foo

if __name__ == "__main__":
    test_input = np.load('test_input_sample.npy')
    results = dtw_cluster_svr(test_input)
    np.save('test_submission_sample.npy', results)
