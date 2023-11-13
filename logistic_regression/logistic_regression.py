import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)

class LogisticRegression:
        
    def __init__(self, learning_rate=0.001, max_iterations=10000):
        '''Initialize variables
        Args:
            learning_rate  : Learning Rate
            max_iterations : Max iterations for training weights
        '''
        self.learning_rate  = learning_rate
        self.max_iterations = max_iterations
        
        self.eps = 1e-7
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        # TODO: Implement
        num_examples = X.shape[0]
        num_features = X.shape[1]
        
        ones = np.ones((num_examples, 1))  

        X = np.hstack((X, ones))
        
        # Initialize weights with appropriate shape
        self.weights = np.zeros(num_features+1)
        
        # Perform gradient ascent
        for i in range(self.max_iterations):
            
            z = np.dot(self.weights.T,X.T)
            
            y_pred = sigmoid(z)
            
            gradient = np.dot(X.T,(y-y_pred))
            
            self.weights = self.weights + self.learning_rate*gradient
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        # TODO: Implement
        num_examples = X.shape[0]
        num_features = X.shape[1]
        
        ones = np.ones((num_examples, 1)) 

        X = np.hstack((X, ones))
        
        z = np.dot(X,self.weights)
        probabilities = z
        probabilities.reshape(probabilities.shape[0],1)
        
        return probabilities
    
    def polynomial_features(self, X, degree):
        """
        Generates polynomial features up to the specified degree for the input feature matrix X.
    
        Parameters:
        - X: Input feature matrix (n_samples, n_features).
        - degree: The degree of polynomial features to generate.
    
        Returns:
        - X_poly: Expanded feature matrix with polynomial features.
        """
    
        n_samples, n_features = X.shape
    
        X_poly = X.copy()
    
        for d in range(2, degree + 1):
            # Generate polynomial features up to the specified degree
            poly_features = np.power(X, d)
            
            # Append the polynomial features to the expanded feature matrix
            X_poly = np.hstack((X_poly, poly_features))
    
        return X_poly
    
    def feature_expansion(self,X):
        res_list = []
            
        for sample in X:
            new_sample = [sample[0], sample[1], sample[0]**2, sample[1]**2, sample[0]*sample[1]]
            res_list.append(new_sample)
        
        res = np.array(res_list)
            
        return res
            

# --- Some utility functions 
def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )

def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

        