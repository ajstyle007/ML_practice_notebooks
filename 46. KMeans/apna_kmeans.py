# decide no.of clusters
# select random centroids
# assign clusters
# move centroids
# check finish

from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
centroids = [(-5,5), (5,5), (-2.5, 2.5), (2.5, -2.5), (-6,-2)]
cluster_std = [1,1,1,1,1]
X, y = make_blobs(n_samples=100, n_features=2, centers=centroids, cluster_std=cluster_std, random_state=23)

import random

class Kmeans:

    def __init__(self, n_clusters=2, max_iter=100):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
    
    def fit_predict(self, X):
        random_index = random.sample(range(0, X.shape[0]), self.n_clusters)
        self.centroids = X[random_index]

        for i in range(self.max_iter):
            # assign clusters 
            cluster_group = self.assign_clusters(X)
            print(cluster_group.shape)
                # move centroids
            old_centrods = self.centroids
            self.centroids = self.move_cenroids(X, cluster_group)
                # check finish
            
            if (old_centrods == self.centroids).all():
                break

        return cluster_group

        #print(self.centroids)
    
    def assign_clusters(self, X):
        cluster_group = []
        distances = []
        
        for row in X:
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot((row-centroid), (row-centroid))))
            min_distance = min(distances)
            index_pos = distances.index(min_distance)
            cluster_group.append(index_pos)
            distances.clear()

        return np.array(cluster_group)
    
    def move_cenroids(self, X, cluster_group):
        new_centroids = []

        cluster_type = np.unique(cluster_group)

        for type in cluster_type:
            new_centroids.append(X[cluster_group == type].mean(axis=0))
        
        return np.array(new_centroids)
    
aj = Kmeans(n_clusters=5, max_iter=100)
y_means = aj.fit_predict(X)

plt.scatter(X[y_means == 0,0], X[y_means == 0,1], color="red")
plt.scatter(X[y_means == 1,0], X[y_means == 1,1], color="green")
plt.scatter(X[y_means == 2,0], X[y_means == 2,1], color="yellow")
plt.scatter(X[y_means == 3,0], X[y_means == 3,1], color="black")
plt.scatter(X[y_means == 4,0], X[y_means == 4,1], color="purple")
plt.show()