# Clustering

Code used to embed and cluster the activity phrases along with sample clustering results with different values of k (for k-means).

## Contents

There are two main steps for the clustering process: embedding phrases and clustering phrases. Each is described here, along with the scripts related to it.

### Embedding Phrases

- `activity_list.csv` : contains a list of all of the activity phrases to be embedded. These have already been convered to present tense to match the input expected by the pretrained models.
- DNT/ : contains code from [Direct Network Transfer](https://arxiv.org/abs/1804.07835) paper which is used to train the activity embedding models, which can then be used to generate embeddings for the activity phrases. *Much of the code in this directory was written/adapted by Li Zhang*.
    - Use last hidden state from these models to get the vector representations for the activities.
    - This directory also includes several pretrained models that use this codebase.

### Clustering Phrases
- `compute_centroids.py` : precomputes the cluster centroids (as vectors) to speed up the process of assigning new activities to predefined clusters in the future.
- `create_plot.py` : creates visualization of 2-D T-SNE reduced version of the clusters.
- `evaluate_clusters.py` : used to compute silhouette, Calinski-Harabaz, and Davies-Bouldin scores for the clusters.
 -`example_clusters/` : some examples of full clustering results using different values of $k$ for k-means clustering given the provided list of activities. Each file is named with the value of $k$.
- `get_similar_centroids.py` : finds clusters that are similar to one another in the embedding space.
- `kmeans.py` : run the actual kmeans clustering.
- `metrics.py` : contains code used by `evaluate_clusters.py` to compute some of the scores.
- `tsne.py` : run the T-SNE projection into 2-D used by `create_plot.py` for visualization.
