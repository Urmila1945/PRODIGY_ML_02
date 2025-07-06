# clustering.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def apply_kmeans(X, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    labels = model.fit_predict(X)
    return model, labels
