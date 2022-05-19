from cluster_svr import cluster_svr
import numpy as np

train, test, truth = random_sample(1200)
test = np.load('test_input.npy')

cluster_labels = np.load('cluster_labels_spi_400.npy')
predictions_dtw = cluster_svr(train, test, cluster_labels)
results = dtw_cluster_svr(test_input)

np.save('test_submission.npy', results)
