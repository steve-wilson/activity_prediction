
import collections
import os
import sys
import numpy as np
import kmeans
import create_plot

def compute_centroids(data_dir, clusters_file_path, out_path):

    ids, X = kmeans.load_matrix(data_dir)
    clusters, set_of_labels = create_plot.load_clusters(clusters_file_path, ids)
    cluster_counts = collections.Counter(clusters)
    cluster_sums = {}
    x_dim_1 = X.shape[1]
    for i in range(len(ids)):
        cluster = clusters[i]
        if cluster not in cluster_sums:
            cluster_sums[cluster] = np.zeros(x_dim_1)
        cluster_sums[cluster] += X[i]

    centroids = {}
    for cluster, cluster_sum in cluster_sums.items():
        centroids[cluster] = cluster_sum / cluster_counts[cluster]

    with open(out_path,'w') as outfile:
        for cluster, centroid in sorted(centroids.items()):
            outfile.write(str(cluster) + ":")
            outfile.write(" ".join([str(x) for x in centroid]) + '\n')

if __name__ == "__main__":
    compute_centroids(sys.argv[1],sys.argv[2],sys.argv[3]) #datadir clusters_file out_file
