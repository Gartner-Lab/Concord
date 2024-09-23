
# clustering.py

import numpy as np
import igraph as ig
import leidenalg
import community as community_louvain
from sklearn.neighbors import kneighbors_graph

class ClusteringModule:
    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors

    def fit(self, embeddings):
        self.embeddings = embeddings
        self._construct_knn_graph()
        self._cluster_embeddings()

    def _construct_knn_graph(self):
        knn_matrix = kneighbors_graph(self.embeddings, self.n_neighbors, include_self=False, mode='connectivity')
        self.graph = ig.Graph.Adjacency(knn_matrix.toarray().tolist())
        self.graph.es['weight'] = knn_matrix.data

    def _cluster_embeddings(self):
        partition = leidenalg.find_partition(self.graph, leidenalg.ModularityVertexPartition)
        self.cluster_labels = np.array(partition.membership)

    def compute_distances(self):
        unique_labels = np.unique(self.cluster_labels)
        centroids = np.array([self.embeddings[self.cluster_labels == label].mean(axis=0) for label in unique_labels])
        self.centroid_distances = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=-1)

    def get_cluster_labels(self):
        return self.cluster_labels

    def get_centroid_distances(self):
        return self.centroid_distances

