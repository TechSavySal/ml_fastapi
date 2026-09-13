import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib


# Training data
X = np.array([
    [22, 30000, 2],
    [25, 35000, 3],
    [28, 40000, 4],
    [32, 50000, 6],
    [35, 60000, 8],
    [40, 70000, 10],
    [45, 80000, 12],
    [50, 90000, 15],
    [55, 95000, 16],
    [60, 100000, 18]
])


# Target
y = np.array([
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0
])


pipeline=Pipeline([
    ("scaler",StandardScaler()),
    ("model",LogisticRegression(max_iter=1000))
])
pipeline.fit(X,y)


# Save model
joblib.dump(pipeline, "model.pkl")

print("Pipeline trained and saved successfully!")
