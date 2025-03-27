import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

class HierarchicalClustering:
    def __init__(self, n_clusters=2, linkage_method='complete'):
        """
        Initialize the Hierarchical Clustering model.
        
        Parameters:
        - n_clusters: Number of clusters to form
        - linkage_method: Linkage criterion ('single', 'complete', 'average', 'ward')
        """
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.labels_ = None
        self.linkage_matrix_ = None

    def fit(self, X):
        """
        Perform hierarchical clustering on the dataset.
        
        Parameters:
        - X: Feature matrix (n_samples, n_features)
        """
        # Compute pairwise distances
        dist_matrix = pdist(X)
        
        # Perform hierarchical clustering
        self.linkage_matrix_ = linkage(dist_matrix, method=self.linkage_method)
        
        # Assign cluster labels
        self.labels_ = fcluster(self.linkage_matrix_, t=self.n_clusters, criterion='maxclust') - 1
    
    def plot_clusters(self, X):
        """
        Scatter plot of clusters.
        
        Parameters:
        - X: Feature matrix (n_samples, n_features)
        """
        if self.labels_ is None:
            raise ValueError("Model has not been fitted yet.")
        
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=self.labels_, cmap='viridis', edgecolors='k')
        plt.title(f'Hierarchical Clustering ({self.linkage_method.capitalize()} Linkage)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

    def plot_dendrogram(self):
        """
        Plot the dendrogram of hierarchical clustering.
        """
        if self.linkage_matrix_ is None:
            raise ValueError("Model has not been fitted yet.")
        
        plt.figure(figsize=(10, 6))
        dendrogram(self.linkage_matrix_)
        plt.title(f'Dendrogram ({self.linkage_method.capitalize()} Linkage)')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()

# Generate sample dataset
np.random.seed(42)
X = np.vstack([
    np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2)),
    np.random.normal(loc=[5, 5], scale=0.5, size=(50, 2)),
    np.random.normal(loc=[0, 5], scale=0.5, size=(50, 2))
])

# Perform clustering
hc = HierarchicalClustering(n_clusters=3, linkage_method='complete')
hc.fit(X)

# Plot results
hc.plot_clusters(X)
hc.plot_dendrogram()
