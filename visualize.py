import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

cluster_labels = np.load('cluster_labels_spi.npy')

plt.imshow(cluster_labels.reshape((13,20)), interpolation='none')
plt.title('DTW clusters')
plt.legend()
plt.show()

dtw_sim_spi = np.load('dtw_sim_spi.npy')
dtw_sim_array = np.array(dtw_sim_spi)
dtw_mean = dtw_sim_array.mean((0,1))

plt.imshow(dtw_mean, interpolation='none')
plt.title('DTW mean distance')
plt.legend()
plt.show()

test = np.load('test_input_sample.npy')
pred = np.load('test_submission_sample.npy')
truth = np.load('truth_sample.npy')

mse = None
meanmse = []
mse_values = []
for t in range(len(test)):
    mse = mean_squared_error(pred[t], truth[t])
    mse_values.append(mse)
    error = pred[t]-truth[t]
    meanmse.append(error.reshape((13, 20)))


plt.imshow(np.array(meanmse).mean(0))#.reshape((13, 20)))
plt.title('MSE SAMPLE X')
plt.show()
#plt.legend()
print('Mean mse : ', np.array(mse_values).mean())