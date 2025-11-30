from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

MODEL_PATH = 'models/ml/classification.pkl'

def train_sample_classifier(X, y):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100))
    ])
    pipe.fit(X, y)
    joblib.dump(pipe, MODEL_PATH)
    return pipe
