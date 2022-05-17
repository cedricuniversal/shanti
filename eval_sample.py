import numpy as np
from sklearn.metrics import mean_squared_error

truth = np.load('truth_sample.npy')
pred = np.load('test_submission_sample_cluster.npy')

print('Predictions shape', pred.shape)
print('Truth shape', truth.shape)
mse_values = []
for i in range(120):
  mse = mean_squared_error(pred[i], truth[i])
  mse_values.append(mse)
print('Mean mse on 120 random samples', np.array(mse_values).mean())
