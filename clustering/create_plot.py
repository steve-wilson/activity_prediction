
import sys
from matplotlib import pyplot as plt

def load_data_file(path):

    i_vec,x_vec, y_vec = [],[],[]
    with open(path) as infile:
        for line in infile.readlines():
            i,x,y = line.strip().split()
            x,y = float(x),float(y)
            x_vec.append(x)
            y_vec.append(y)
            i_vec.append(i)

    return i_vec,x_vec,y_vec

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

def create_plot(data_path, out_path, clusters_path=""):

    ids,x,y = load_data_file(data_path)
    labels = []
    num_colors = 1
    if clusters_path:
        labels, set_of_labels = load_clusters(clusters_path, ids)
        num_colors = len(set_of_labels)
    else:
        labels = [0] * len(ids)
        set_of_labels = [0]

    plt.scatter(x,y, c=labels, marker=',', s=.2, cmap=plt.cm.get_cmap("gnuplot2", num_colors))
    plt.figure(num=1, figsize=(2,5), dpi=100)
    plt.savefig(out_path, bbox_inches='tight', format='png', dpi=1000)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        create_plot(sys.argv[1],sys.argv[2])
    elif len(sys.argv) == 4:
        create_plot(sys.argv[1],sys.argv[2], sys.argv[3])
    else:
        print("Requires arguments: data_path out_path.png [clusters_path]")
