from sklearn.cluster import KMeans
import joblib

MODEL_PATH = 'models/ml/clustering.pkl'

def train_kmeans(X, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    joblib.dump(kmeans, MODEL_PATH)
    return kmeans
