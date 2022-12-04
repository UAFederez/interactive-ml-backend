from flask import Flask, request
from flask_restful import Api, Resource
import json
import numpy as np

class KMeansClustering(Resource):
    def compute_cluster_assingments(self, X_data, centroids):
        n_clusters = centroids.shape[0]
        distances  = (np.tile(X_data, n_clusters) - centroids.reshape(1, -1)) ** 2
        distances  = distances.reshape((distances.shape[0], n_clusters, -1))
        distances  = np.sum(distances, axis = 2)

        return np.argmin(distances, axis = 1)

    def fit_kmeans_cluster(self, K, X_data, num_iter = None, min_avg_update_dist = None, random_seed = None):
        random    = np.random.RandomState(random_seed)
        # Sample k points without replacement to initalize the cluster centroids
        centroids = X_data[random.choice(np.arange(0, X_data.shape[0]), K, replace = False)] + random.randn(K, X_data.shape[1])
        cent_hist = []

        curr_iter = 0

        while True:
            assignments = self.compute_cluster_assingments(X_data, centroids)
            prev_cent   = centroids.copy()
            cent_hist.append(prev_cent)

            for cluster_idx in range(K):
                members = X_data[assignments == cluster_idx]
                if members.shape[0] == 0:
                    new_loc = X_data[random.choice(np.arange(0, X_data.shape[0]), 1, replace = False)]
                else:
                    new_loc = np.mean(members, axis = 0, keepdims = True)
                centroids[cluster_idx, :] = new_loc

            curr_iter += 1

            # Either stop based on num_iter or if centroids update less than min_update_dist
            if num_iter:
                stop_condition = curr_iter >= num_iter 
            elif min_avg_update_dist:
                dist_from_prev = np.sqrt(np.sum((centroids - prev_cent) ** 2, axis = 1))
                mean_dist_prev = np.mean(dist_from_prev)
                stop_condition = mean_dist_prev < min_avg_update_dist

            if stop_condition:
                break

        return centroids, cent_hist

    
    def post(self):
        data = request.json
        train_x    = np.array(data['train_x']).T
        n_clusters = data['n_clusters']

        max_num_iter = data['max_num_iter'] if 'max_num_iter' in data else None
        min_avg_update_dist = data['min_avg_update_dist'] if 'min_avg_update_dist' in data else None

        centroids, hist = self.fit_kmeans_cluster(n_clusters, train_x, 
                                                  num_iter = max_num_iter,
                                                  min_avg_update_dist = min_avg_update_dist)

        response = {
            'status'   : 'success',
            'centroids': centroids.tolist(),
            'history'  : {
                'centroids': [ centroid.tolist() for centroid in hist ],
            }
        }

        return response
