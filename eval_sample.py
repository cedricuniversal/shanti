import numpy as np
from sklearn.metrics import mean_squared_error

truth = np.load('truth_sample.npy')
pred = np.load('test_submission_sample_cluster.npy')
#pred = np.load('test_submission_sample_alone.npy')

#print('Predictions shape', pred.shape)
#print('Truth shape', truth.shape)

try:
  mse_values = list(np.load('mse_values.npy'))
except:
  mse_values = []

for i in range(120):
  mse = mean_squared_error(pred[i], truth[i])
  mse_values.append(mse)
  #print('SAMPLE SVR Mean mse on %i random samples: '%len(mse_values), np.array(mse_values).mean())
print('MEAN DTWSVR mse on %i random samples: '%len(mse_values), np.array(mse_values).mean())
np.save('mse_values.npy', mse_values)