from dtw_cluster_sample import dtw_cluster_svr
import numpy as np

test = np.load('test_input.npy')

results = dtw_cluster_svr(test)

np.save('test_submission.npy', results)
