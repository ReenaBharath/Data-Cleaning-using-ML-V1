"""
Machine learning model functions for data analysis.

This module provides a set of functions for applying machine learning techniques to data analysis,
including data preparation, clustering, and anomaly detection. It leverages scikit-learn's
implementation of various algorithms such as KMeans clustering, PCA for dimensionality reduction,
and Isolation Forest for anomaly detection.

The module is organized into logical sections:
1. Data Preparation Functions - For preparing numeric data for ML models
2. Clustering Functions - For identifying natural groupings in the data
3. Anomaly Detection Functions - For identifying outliers and unusual patterns

These functions can be used individually or combined to create a comprehensive
data analysis pipeline for data cleaning and exploration.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import warnings
from src.utils.utils import ensure_dir

def prepare_numeric_data(df):
    """
    Prepare numeric data for ML models.
    
    This function extracts and preprocesses numeric columns from a DataFrame to make them
    suitable for machine learning algorithms. It performs the following steps:
    1. Selects only numeric columns
    2. Removes columns with excessive missing values (>50% by default)
    3. Fills remaining missing values with column means
    
    Args:
        df (pandas.DataFrame): Input DataFrame containing mixed data types
        
    Returns:
        pandas.DataFrame: Preprocessed DataFrame containing only numeric columns with no missing values
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Drop columns with too many missing values
    threshold = 0.5  # 50% missing values
    columns_before = set(numeric_df.columns)
    numeric_df = numeric_df.dropna(axis=1, thresh=int(threshold * len(numeric_df)))
    columns_after = set(numeric_df.columns)
    
    # Report dropped columns
    dropped_columns = columns_before - columns_after
    if dropped_columns:
        print(f"Warning: Dropped {len(dropped_columns)} columns due to excessive missing values: {', '.join(dropped_columns)}")
    
    # Fill remaining missing values with column means
    # Added note about potential data integrity concerns
    print("Note: Filling missing numeric values with column means. This may affect data integrity.")
    numeric_df = numeric_df.fillna(numeric_df.mean())
    
    return numeric_df

def prepare_data_for_clustering(df, n_components=2, random_state=42):
    """
    Prepare data for clustering using PCA.
    
    This function prepares data for clustering by:
    1. Extracting and preprocessing numeric data
    2. Standardizing features to have zero mean and unit variance
    3. Reducing dimensionality using Principal Component Analysis (PCA)
    
    PCA is used to reduce the dimensionality of the data while preserving
    as much variance as possible, making it easier to visualize and cluster.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        n_components (int): Number of principal components to keep
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_scaled, X_pca) where:
            - X_scaled is the standardized data
            - X_pca is the PCA-transformed data
        Returns (None, None) if there's an error or insufficient data
    """
    try:
        # Prepare numeric data
        X = prepare_numeric_data(df)
        
        if len(X.columns) < 2:
            print("Not enough numeric columns for clustering")
            return None, None
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(n_components, len(X.columns)), random_state=random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        return X_scaled, X_pca
    except ImportError:
        print("scikit-learn not available for clustering")
        return None, None
    except Exception as e:
        print(f"Error preparing data for clustering: {e}")
        return None, None

def train_kmeans(X, n_clusters=5, random_state=42):
    """
    Train KMeans clustering model.
    
    KMeans clustering partitions the data into k clusters, where each observation
    belongs to the cluster with the nearest mean. This implementation uses the
    scikit-learn KMeans algorithm with the following steps:
    1. Initialize cluster centers (using k-means++ by default)
    2. Assign each data point to the nearest cluster
    3. Update cluster centers based on assigned points
    4. Repeat steps 2-3 until convergence
    
    Args:
        X (numpy.ndarray): Input data matrix, typically standardized
        n_clusters (int): Number of clusters to form
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (kmeans_model, cluster_labels) where:
            - kmeans_model is the trained KMeans model
            - cluster_labels are the assigned cluster indices for each data point
        Returns (None, None) if there's an error
    """
    try:
        # Train KMeans model
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = kmeans.fit_predict(X)
        
        return kmeans, clusters
    except ImportError:
        print("scikit-learn not available for KMeans clustering")
        return None, None
    except Exception as e:
        print(f"Error training KMeans model: {e}")
        return None, None

def train_clustering(df, n_clusters=5, random_state=42):
    """
    Train a clustering model on the dataframe.
    
    This is a high-level function that combines data preparation, standardization,
    dimensionality reduction, and KMeans clustering into a single workflow.
    It performs the following steps:
    1. Extract and preprocess numeric data
    2. Standardize features
    3. Reduce dimensionality to 2 components for visualization
    4. Apply KMeans clustering
    5. Report cluster distribution
    
    Args:
        df (pandas.DataFrame): DataFrame to cluster
        n_clusters (int): Number of clusters to form
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (kmeans_model, X_pca, cluster_labels) where:
            - kmeans_model is the trained KMeans model
            - X_pca is the PCA-transformed data for visualization
            - cluster_labels are the assigned cluster indices
        Returns (None, None, None) if there's an error or insufficient data
    """
    # Get numeric data
    X = prepare_numeric_data(df)
    
    if X is None or X.shape[0] < n_clusters:
        print("Not enough numeric data for clustering")
        return None, None, None
    
    try:
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Reduce dimensionality for visualization
        pca = PCA(n_components=2, random_state=random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        # Use the train_kmeans function instead of duplicating code
        kmeans, clusters = train_kmeans(X_scaled, n_clusters=n_clusters, random_state=random_state)
        
        if kmeans is None:
            return None, None, None
        
        print(f"Clustering completed with {n_clusters} clusters")
        print(f"Cluster distribution: {pd.Series(clusters).value_counts().to_dict()}")
        
        return kmeans, X_pca, clusters
    except Exception as e:
        print(f"Error in clustering: {e}")
        return None, None, None

def train_isolation_forest(X, contamination=0.05, random_state=42):
    """
    Train Isolation Forest model for anomaly detection.
    
    Isolation Forest is an algorithm for anomaly detection that works by isolating
    observations by randomly selecting a feature and then randomly selecting a split
    value between the maximum and minimum values of that feature. The logic is that
    anomalies are few and different, so they should be isolated earlier in the forest.
    
    This function:
    1. Trains an Isolation Forest model on the input data
    2. Predicts anomalies (-1 for anomalies, 1 for normal points)
    3. Calculates anomaly scores (higher values indicate more anomalous points)
    
    Args:
        X (numpy.ndarray): Input data matrix, typically standardized
        contamination (float): Expected proportion of anomalies in the data (0.0 to 0.5)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (model, is_anomaly, anomaly_scores) where:
            - model is the trained Isolation Forest model
            - is_anomaly is a boolean mask (True for anomalies)
            - anomaly_scores are the anomaly scores (higher = more anomalous)
        Returns (None, None, None) if there's an error
    """
    try:
        # Train Isolation Forest model
        model = IsolationForest(contamination=contamination, random_state=random_state)
        model.fit(X)
        
        # Predict anomalies (-1 for anomalies, 1 for normal)
        anomalies = model.predict(X)
        
        # Convert to boolean mask (True for anomalies)
        is_anomaly = anomalies == -1
        
        # Calculate anomaly scores
        anomaly_scores = model.decision_function(X)
        # Invert scores so higher values indicate more anomalous
        anomaly_scores = -anomaly_scores
        
        return model, is_anomaly, anomaly_scores
    except ImportError:
        print("scikit-learn not available for anomaly detection")
        return None, None, None
    except Exception as e:
        print(f"Error training Isolation Forest model: {e}")
        return None, None, None

def train_anomaly_detection(df, random_state=42):
    """
    Train anomaly detection model and return results.
    
    This is a high-level function that combines data preparation and
    Isolation Forest anomaly detection into a single workflow. It:
    1. Extracts and preprocesses numeric data
    2. Applies Isolation Forest for anomaly detection
    3. Returns the model and results
    
    Anomaly detection is useful for identifying outliers and unusual patterns
    in the data that may represent errors, fraud, or other interesting events.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (model, is_anomaly, anomaly_scores) where:
            - model is the trained Isolation Forest model
            - is_anomaly is a boolean mask (True for anomalies)
            - anomaly_scores are the anomaly scores (higher = more anomalous)
        Returns (None, None, None) if there's an error or insufficient data
    """
    # Prepare numeric data
    X = prepare_numeric_data(df)
    
    if len(X.columns) < 2:
        print("Not enough numeric columns for anomaly detection")
        return None, None, None
    
    # Train Isolation Forest model
    model, is_anomaly, anomaly_scores = train_isolation_forest(X, random_state=random_state)
    
    if model is None:
        return None, None, None
    
    return model, is_anomaly, anomaly_scores
