from dtw_cluster_svr import dtw_cluster_svr
import numpy as np

test_input = np.load('test_input.npy')

results = dtw_cluster_svr(test_input)

np.save('test_submission.npy', results)
