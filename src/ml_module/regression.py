import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = 'models/ml/regression.pkl'

def train_sample_regression(X, y):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=50))
    ])
    pipe.fit(X, y)
    joblib.dump(pipe, MODEL_PATH)
    return pipe

def predict_regression_sample(X):
    try:
        model = joblib.load(MODEL_PATH)
        pred = model.predict(X)
        return pred.tolist()
    except Exception as e:
        # placeholder behavior if model not present
        return [0.0]
