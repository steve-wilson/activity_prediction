
import os
import sys

from MulticoreTSNE import MulticoreTSNE as TSNE
#from sklearn.manifold import TSNE
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

def run_tsne(path):

    ids, X = load_matrix(path)
    tsne = TSNE(n_jobs=8)
#    tsne = TSNE(metric='cosine')
    Y = tsne.fit_transform(X)
    for i,row in enumerate(Y):
        print(' '.join([ids[i],str(row[0]),str(row[1])]))

if __name__ == "__main__":
    run_tsne(sys.argv[1]) # matrix files dir
