
import collections
import os
import sys

import numpy as np
from scipy.spatial.distance import cosine

def load_centroids(path):

    centroids = {}

    with open(path) as cfile:
        for line in cfile:
            centroid_id,vector = line.strip().split(':')
            centroids[int(float(centroid_id))] = np.array([float(x) for x in vector.split()])

    return centroids

def get_most_similar_centroids(centroid_path, out_file_path):

    distances = collections.defaultdict(dict)
    centroids = load_centroids(centroid_path)
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            if i<=j:
                distance = cosine(centroids[i],centroids[j])
                distances[i][j] = distance

    with open(out_file_path,'w') as outfile:
        for i,distances_to_j in distances.items():
            ranking = sorted([(j,d) for j,d in distances_to_j.items()], key=lambda x:x[1])
            outfile.write(str(i) + ':')
            outfile.write(' '.join([str(r[0])+','+str(r[1]) for r in ranking]) +'\n')

if __name__ == "__main__":
    get_most_similar_centroids(sys.argv[1], sys.argv[2]) #centroids_path, out_file_path
