import numpy as np
from sklearn.metrics import mean_squared_error

truth = np.load('truth_sample.npy')
pred_svr = np.load('test_submission_sample_svr.npy')
pred_cluster = np.load('test_submission_sample_cluster.npy')


try:
  mse_values_cluster = list(np.load('mse_values_cluster.npy'))
  mse_values_svr = list(np.load('mse_values_svr.npy'))
except:
  mse_values_cluster = []
  mse_values_svr = []

for i in range(120):
  mse = mean_squared_error(pred_cluster[i], truth[i])
  mse_values_cluster.append(mse)
  mse = mean_squared_error(pred_svr[i], truth[i])
  mse_values_svr.append(mse)
  #print('SAMPLE SVR Mean mse on %i random samples: '%len(mse_values), np.array(mse_values).mean())
print('MEAN CLUSTER mse on %i random samples: '%len(mse_values_cluster), np.array(mse_values_cluster).mean())
print('MEAN SVR mse on %i random samples: '%len(mse_values_svr), np.array(mse_values_svr).mean())
np.save('mse_values_cluster.npy', mse_values_cluster)
np.save('mse_values_svr.npy', mse_values_svr)
