import numpy as np
import matplotlib.pyplot as plt

cluster_labels = np.load('cluster_labels_spi.npy')

plt.imshow(cluster_labels.reshape((13,20)), interpolation='none')
plt.title('DTW clusters')
plt.show()

dtw_sim_spi = np.load('dtw_sim_spi.npy')
dtw_sim_array = np.array(dtw_sim_spi)
dtw_mean = dtw_sim_array.mean((0,1))

plt.imshow(dtw_mean, interpolation='none')
plt.title('DTW mean distance')
plt.show()

