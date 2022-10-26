from utils import *
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error

train_data, train_labels = parseFileWithLabel(TRAIN_FILE_PATH)
validation_data, validation_labels = parseFileWithLabel(VALIDATION_FILE_PATH)
indexes_test, test_data = parseFileWithoutLabel(TEST_FILE_PATH)

# split into the regions (used number of clusters = 5)
kmeans = KMeans(n_clusters=5, n_init=1000, n_jobs=32, max_iter=5000, algorithm="auto").fit(np.concatenate([train_labels, validation_labels]))
print("size = {} => label = {}".format(len(kmeans.labels_), kmeans.labels_))
print("size = {} => label = {}".format(len(kmeans.cluster_centers_), kmeans.cluster_centers_))

dictionary = {}
dictionary["cluster_label_train"] = kmeans.labels_[:len(train_labels)]
dictionary["cluster_label_validation"] = kmeans.labels_[len(train_labels):]
dictionary["clusters_centroids"] = kmeans.cluster_centers_

# save the clustering info (to be used in CharCNN as classification labels)
with open(KMEANS_INFO_PATH, 'wb') as f:
    np.savez(f, **dictionary)

# maximum mae and mse for this clustering on validation set
mae = mean_absolute_error(validation_labels, [kmeans.cluster_centers_[cluster] for cluster in kmeans.labels_[len(train_labels):]])
mse = mean_squared_error(validation_labels, [kmeans.cluster_centers_[cluster] for cluster in kmeans.labels_[len(train_labels):]])
print("Maximum mae for this particular clustering: {}".format(mae))
print("Maximum mse for this particular clustering: {}".format(mse))











