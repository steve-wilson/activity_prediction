
import os
import sys

import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from metrics import davies_bouldin_score

def load_matrix(path, normalize=True):
    vectors = []
    ids = []
    norms = []
    for f in os.listdir(path):
        with open(path + f) as vectors_file:
            for line in vectors_file.readlines():
                parts = line.strip().split('\t')
                ids.append(parts[0].strip())
                vectors.append([float(x) for x in parts[1].split()])
                if normalize:
                    norms.append(float(parts[2]))
    if normalize:
        return ids, np.array(vectors,dtype=np.float16)/np.array(norms,dtype=np.float16)[:,None]
    else:
        return ids, np.array(vectors,dtype=np.float16)

def load_clusters(path, ids):

    labels = []
    seen_clusters = set([])
    id2cluster = {}
    with open(path) as infile:
        for line in infile.readlines():
            i,l = line.strip().split()
            seen_clusters.add(l)
            id2cluster[i] = float(l)
    for i in ids:
        labels.append(id2cluster[i])

    return labels, sorted(list(seen_clusters))

def evaluate_clusters(X, ids, labels_file):

    print("Evaluating: "+ labels_file)
    clusters, label_list = load_clusters(labels_file, ids)

    # run evaluations
    # 1. Silhouette Coeffient
    sc = 0.0
    for i in range(100):
        sc += silhouette_score(X, clusters, sample_size=1000)
    sc /= 100.0

    # 2. Variance Ratio Criterion
    vrc = calinski_harabaz_score(X, clusters)

    # 3. 
    dbs = davies_bouldin_score(X, clusters)

    print("Silhouette, Calinski-Harabaz, Davies-Bouldin")
    print([sc,vrc,dbs])

def run_evaluations(data_dir, cluster_results_dir):

    ids,X = load_matrix(data_dir)
    for result in os.listdir(cluster_results_dir):
        evaluate_clusters(X, ids, cluster_results_dir + os.sep + result)

if __name__ == "__main__":
    run_evaluations(sys.argv[1], sys.argv[2]) # vectors_dir, clustering_results_dir
