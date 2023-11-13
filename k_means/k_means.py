import numpy as np 
import pandas as pd 
import sys
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        pass
        
    def fit(self, X, k):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        #X = X.values
        best_centroids = []
        best_score = float('-inf')
        best_distorsion = float('inf')
           
        means = []
        classes = []


        #centroid inizialization
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        # Select the first k shuffled indices as initial centroids
        initial_indices = indices[:k]

        # Use the selected indices to extract the corresponding data points as centroids
        self.centroids = [X[i, :] for i in initial_indices]

        # Initialize centroids with random data points from X
        # Make sure to implement this part
        means = [centroid.copy() for centroid in self.centroids]

        while True:
            self.centroids = means.copy()
            classes = []

            for i in range(X.shape[0]):
                ed_min = float('inf')
                ed_index = 0

                for j in range(len(self.centroids)):
                    ed = euclidean_distance(X[i, :], self.centroids[j])
                    if ed < ed_min:
                        ed_min = ed
                        ed_index = j

                classes.append(ed_index)

            for i in range(len(means)):
                mask = [element == i for element in classes]
                selected_elements = X[mask, :]
                if len(selected_elements) > 0:
                    means[i] = np.mean(selected_elements, axis=0)   

            if np.array_equal(means, self.centroids):
                break
            
        return 0
            
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        # TODO: Implement 
        #X = X.values
        cluster_assignments = []
        
        for i in range(X.shape[0]):
            ed_min = float('inf')
            ed_index = 0
            
            for j in range(len(self.centroids)):
                ed = euclidean_distance(X[i, :], self.centroids[j])
                if ed < ed_min:
                    ed_min = ed
                    ed_index = j
                    
            cluster_assignments.append(ed_index)
            
        return cluster_assignments   
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return np.vstack(self.centroids)
    
    def standardize(self, X):
        data = X.values
        mean_dim1 = np.mean(data[:, 0])
        std_dim1 = np.std(data[:, 0])
        
        mean_dim2 = np.mean(data[:, 1])
        std_dim2 = np.std(data[:, 1])
        
        # Standardizzazione delle feature
        data_standardized = np.copy(data)
        data_standardized[:, 0] = (data[:, 0] - mean_dim1) / std_dim1
        data_standardized[:, 1] = (data[:, 1] - mean_dim2) / std_dim2
    
        return data_standardized
    
# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum(axis=1).sum(axis=0) #modified by me ading sum(0)
        
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
  