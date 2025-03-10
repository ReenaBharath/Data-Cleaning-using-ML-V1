"""
Machine learning model functions for data analysis.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import warnings
from src.utils.utils import ensure_dir

# Suppress warnings
warnings.filterwarnings('ignore')

def prepare_numeric_data(df):
    """Prepare numeric data for ML models."""
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Drop columns with too many missing values
    threshold = 0.5  # 50% missing values
    numeric_df = numeric_df.dropna(axis=1, thresh=int(threshold * len(numeric_df)))
    
    # Fill remaining missing values with column means
    numeric_df = numeric_df.fillna(numeric_df.mean())
    
    return numeric_df

def prepare_data_for_clustering(df, n_components=2):
    """Prepare data for clustering using PCA."""
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
        pca = PCA(n_components=min(n_components, len(X.columns)))
        X_pca = pca.fit_transform(X_scaled)
        
        return X_scaled, X_pca
    except ImportError:
        print("scikit-learn not available for clustering")
        return None, None
    except Exception as e:
        print(f"Error preparing data for clustering: {e}")
        return None, None

def train_kmeans(X, n_clusters=5):
    """Train KMeans clustering model."""
    try:
        # Train KMeans model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
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
    
    Args:
        df (pandas.DataFrame): DataFrame to cluster
        n_clusters (int): Number of clusters
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (model, X_pca, clusters)
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
        
        # Train KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = kmeans.fit_predict(X_scaled)
        
        print(f"Clustering completed with {n_clusters} clusters")
        print(f"Cluster distribution: {pd.Series(clusters).value_counts().to_dict()}")
        
        return kmeans, X_pca, clusters
    except Exception as e:
        print(f"Error in clustering: {e}")
        return None, None, None

def train_isolation_forest(X, contamination=0.05):
    """Train Isolation Forest model for anomaly detection."""
    try:
        # Train Isolation Forest model
        model = IsolationForest(contamination=contamination, random_state=42)
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

def train_anomaly_detection(df):
    """Train anomaly detection model and return results."""
    # Prepare numeric data
    X = prepare_numeric_data(df)
    
    if len(X.columns) < 2:
        print("Not enough numeric columns for anomaly detection")
        return None, None, None
    
    # Train Isolation Forest model
    model, is_anomaly, anomaly_scores = train_isolation_forest(X)
    
    if model is None:
        return None, None, None
    
    return model, is_anomaly, anomaly_scores
