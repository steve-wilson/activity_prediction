
import os
import sys

from sklearn.cluster import MiniBatchKMeans
import numpy as np

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
        return ids, np.array(vectors)/np.array(norms)[:,None]
    else:
        return ids, np.array(vectors)

def run_kmeans(path, out_dir):

    ids, X = load_matrix(path)
    k_list = [2**n for n in [12,13]]
    for k in k_list:
        kmeans = MiniBatchKMeans(k, n_init=10)
        kmeans.fit(X)
        with open(out_dir + os.sep + "kmeans_clusters_" + str(k) + ".out",'w') as outfile:
            for i,label in enumerate(kmeans.labels_):
                outfile.write(' '.join([ids[i],str(label)]) + '\n')
        print("k="+str(k))
        print("inertia="+str(kmeans.inertia_))
        print()

if __name__ == "__main__":
    run_kmeans(sys.argv[1], sys.argv[2]) # matrix files dir, output dir
